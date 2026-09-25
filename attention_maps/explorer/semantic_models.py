"""Optional pretrained encoders. No model imports or downloads at UI import time."""

import os
from pathlib import Path

import numpy as np

MODEL_PRESETS = {
    "nepali-bert": {"id": "Shushant/nepaliBERT", "backend": "bert", "role": "Nepali MLM baseline"},
    "nepberta": {"id": "NepBERTa/NepBERTa", "backend": "tensorflow", "role": "Nepali MLM baseline"},
    "embeddinggemma": {"id": "google/embeddinggemma-300m", "backend": "sentence_transformers", "role": "Candidate teacher; Nepali superiority unverified"},
}
PROMPTS = {"clustering": "task: clustering | query: ", "similarity": "task: sentence similarity | query: "}


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Encoder returned invalid embeddings")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Encoder returned a zero embedding")
    return values / norms


def text_chunks(tokenizer, text: str, limit: int, prompt: str = ""):
    """Cover every source token; recheck decoded chunks to prevent silent truncation."""
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    reserve = len(tokenizer.encode(prompt, add_special_tokens=True)) + 2
    budget = limit - reserve
    if budget <= 0:
        raise ValueError("max_length is too small for the model prompt and special tokens")
    start = 0
    while start < len(ids):
        stop = min(len(ids), start + budget)
        while True:
            chunk = tokenizer.decode(ids[start:stop], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            length = len(tokenizer.encode(prompt + chunk, add_special_tokens=True, truncation=False))
            if length <= limit:
                break
            if stop - start <= 1:
                raise ValueError("A single token cannot fit in max_length after decoding")
            stop = start + max(1, (stop - start) // 2)
        yield chunk, stop - start
        start = stop


class DocumentEncoder:
    """Token-weighted mean of chunk embeddings, then L2 normalization per record."""

    def __init__(self, args, token=None):
        preset = MODEL_PRESETS[args.model]
        model_id = args.model_id or preset["id"]
        backend = preset["backend"]
        revision = args.model_revision
        if not Path(model_id).is_dir():
            from huggingface_hub import HfApi
            revision = HfApi(token=token).model_info(model_id, revision=revision).sha
        kwargs = dict(revision=revision, token=token, trust_remote_code=False)
        self.prompt = PROMPTS[args.embedding_task] if backend == "sentence_transformers" else ""
        self.batch_size = args.batch_size
        self.backend = backend
        if backend == "tensorflow":
            # Keep original NepBERTa runnable in its separate Transformers 4 environment.
            os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
            os.environ.setdefault("USE_TF", "1")
            try:
                import tensorflow as tf
                from transformers import AutoTokenizer, TFAutoModel
            except ImportError as error:
                raise RuntimeError("Original NepBERTa needs the separate requirements-nepberta.txt environment (Transformers <5). See docs/scripts/cluster-dataset.md.") from error
            if args.device == "cpu":
                tf.config.set_visible_devices([], "GPU")
            elif args.device == "cuda" and not tf.config.list_physical_devices("GPU"):
                raise ValueError("CUDA requested, but TensorFlow found no GPU")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
            self.model = TFAutoModel.from_pretrained(model_id, **kwargs)
            self.tf = tf
            native_limit = self.model.config.max_position_embeddings
            self.dimension = self.model.config.hidden_size
            device = "tensorflow automatic" if args.device != "cpu" else "cpu"
        else:
            import torch
            self.torch = torch
            device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
            if device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA requested, but PyTorch found no GPU")
            self.device = device
            if backend == "sentence_transformers":
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(model_id, device=device, model_kwargs={"torch_dtype": torch.float32}, **kwargs)
                self.tokenizer = self.model.tokenizer
                native_limit = self.model.max_seq_length
                self.dimension = self.model.get_sentence_embedding_dimension()
            else:
                from transformers import AutoModel, AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
                self.model = AutoModel.from_pretrained(model_id, **kwargs).to(device).eval()
                native_limit = self.model.config.max_position_embeddings
                self.dimension = self.model.config.hidden_size
        self.max_length = args.max_length or min(native_limit, 2048)
        if self.max_length > native_limit or self.max_length < 16:
            raise ValueError(f"max_length must be between 16 and the model limit {native_limit}")
        if backend == "sentence_transformers":
            self.model.max_seq_length = self.max_length
        self.metadata = {"preset": args.model, "id": model_id, "revision": revision, "backend": backend,
                         "role": preset["role"], "dimension": self.dimension, "device": device,
                         "task": args.embedding_task, "prompt": self.prompt, "max_length": self.max_length,
                         "pooling": "attention-masked mean excluding special tokens" if backend != "sentence_transformers" else "published SentenceTransformer pipeline",
                         "long_records": "all token chunks; token-weighted mean of normalized chunk vectors; L2-normalized document",
                         "dtype": "float32"}

    def _encode_chunks(self, texts):
        if self.backend == "sentence_transformers":
            return normalize_rows(self.model.encode(texts, prompt=self.prompt, batch_size=self.batch_size,
                                  normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False))
        tensors = "tf" if self.backend == "tensorflow" else "pt"
        inputs = self.tokenizer(texts, padding=True, truncation=False, return_tensors=tensors, return_special_tokens_mask=True)
        special = inputs.pop("special_tokens_mask")
        if self.backend == "tensorflow":
            tf = self.tf
            hidden = self.model(**inputs, training=False).last_hidden_state
            mask = tf.cast(inputs["attention_mask"] * (1 - special), hidden.dtype)[..., None]
            values = tf.reduce_sum(hidden * mask, axis=1) / tf.maximum(tf.reduce_sum(mask, axis=1), 1)
            return normalize_rows(values.numpy())
        torch = self.torch
        special = special.to(self.device)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            mask = (inputs["attention_mask"] * (1 - special)).unsqueeze(-1).to(hidden.dtype)
            values = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return normalize_rows(values.float().cpu().numpy())

    def encode(self, texts):
        sums = np.zeros((len(texts), self.dimension), dtype=np.float64)
        counts = np.zeros(len(texts), dtype=np.int64)
        batch, destinations = [], []

        def flush():
            vectors = self._encode_chunks(batch)
            for vector, (row, weight) in zip(vectors, destinations):
                sums[row] += vector * weight
                counts[row] += 1
            batch.clear()
            destinations.clear()

        for row, text in enumerate(texts):
            for chunk, weight in text_chunks(self.tokenizer, text, self.max_length, self.prompt):
                batch.append(chunk)
                destinations.append((row, weight))
                if len(batch) == self.batch_size:
                    flush()
        if batch:
            flush()
        return normalize_rows(sums), counts
