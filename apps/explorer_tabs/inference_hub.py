"""One focused entry point for the explorer's inference and evaluation tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def render_inference_hub(
    *,
    st: Any,
    inventory: dict[str, Any],
    spec: Any,
    cached_sample: Any,
) -> None:
    """Render exactly one selected inference workflow at a time."""

    st.markdown("### Inference")
    st.caption(
        "Use these tools for model inspection and evaluation. They are separate "
        "from preprocessing EDA and never unlock or modify cleaning stages."
    )
    mode = st.selectbox(
        "Inference tool",
        (
            "Model comparison",
            "Local base vs finetuned",
            "Translation evaluation",
            "Decoder benchmarks",
        ),
        key="focused-inference-tool",
    )

    cached = _cached_services(st)
    if mode == "Model comparison":
        from .comparison import render_comparison_tab

        render_comparison_tab(
            st=st,
            inventory=inventory,
            spec=spec,
            cached_sample=cached_sample,
            cached_arkios=cached["arkios"],
            cached_gemma4_base=cached["gemma4_base"],
            cached_google_backend=cached["google_backend"],
            cached_himalayagpt=cached["himalayagpt"],
            cached_huggingface_backend=cached["huggingface_backend"],
            cached_iriis_gpt2=cached["iriis_gpt2"],
            cached_local_model_pair=cached["local_model_pair"],
        )
    elif mode == "Local base vs finetuned":
        from .local_models import render_local_inference_tab

        render_local_inference_tab(
            st=st,
            inventory=inventory,
            spec=spec,
            cached_sample=cached_sample,
            cached_local_model_pair=cached["local_model_pair"],
        )
    elif mode == "Translation evaluation":
        from .evaluation import render_evaluation_tab

        render_evaluation_tab(
            st=st,
            cached_arkios=cached["arkios"],
            cached_flores_examples=cached["flores_examples"],
            cached_gemma4_base=cached["gemma4_base"],
            cached_google_backend=cached["google_backend"],
            cached_himalayagpt=cached["himalayagpt"],
            cached_iriis_gpt2=cached["iriis_gpt2"],
            cached_lima_teacher=cached["lima_teacher"],
            cached_local_model_pair=cached["local_model_pair"],
        )
    else:
        from .evaluation import render_nlue_tab

        render_nlue_tab(
            st=st,
            cached_arkios=cached["arkios"],
            cached_gemma4_base=cached["gemma4_base"],
            cached_google_backend=cached["google_backend"],
            cached_himalayagpt=cached["himalayagpt"],
            cached_iriis_gpt2=cached["iriis_gpt2"],
            cached_local_model_pair=cached["local_model_pair"],
            cached_nlue_examples=cached["nlue_examples"],
        )


def _cached_services(st: Any) -> dict[str, Any]:
    """Create lazy cached adapters without loading model modules at app startup."""

    @st.cache_data(show_spinner=False)
    def cached_flores_examples(
        split: str,
        offset: int,
        limit: int,
        credential_fingerprint: str,
        _token: str,
    ) -> list[Any]:
        del credential_fingerprint
        from attention_maps.evaluation.flores import load_flores_examples

        return load_flores_examples(
            split=split,
            offset=offset,
            limit=limit,
            token=_token or None,
        )

    @st.cache_data(show_spinner=False)
    def cached_nlue_examples(
        task_key: str,
        offset: int,
        limit: int,
        credential_fingerprint: str,
        _token: str,
    ) -> list[Any]:
        del credential_fingerprint
        from attention_maps.evaluation.nlue import load_nlue_examples

        return load_nlue_examples(
            task_key,
            offset=offset,
            limit=limit,
            token=_token or None,
        )

    @st.cache_resource(show_spinner=False)
    def cached_google_backend(
        model_id: str,
        credential_fingerprint: str,
        _api_key: str,
    ) -> Any:
        del credential_fingerprint
        from attention_maps.inference.comparison import GoogleGenAIBackend

        return GoogleGenAIBackend(model_id, _api_key)

    @st.cache_resource(show_spinner=False)
    def cached_lima_teacher(
        model_id: str,
        credential_fingerprint: str,
        _api_key: str,
    ) -> Any:
        del credential_fingerprint
        from attention_maps.inference.gemini_translation import (
            GeminiTranslationBackend,
        )

        return GeminiTranslationBackend(_api_key, model_id)

    @st.cache_resource(show_spinner=False)
    def cached_huggingface_backend(
        model_id: str,
        provider: str,
        credential_fingerprint: str,
        _token: str,
    ) -> Any:
        del credential_fingerprint
        from attention_maps.inference.comparison import HuggingFaceBackend

        return HuggingFaceBackend(model_id, _token, provider)

    @st.cache_resource(show_spinner=False)
    def cached_local_model_pair(
        adapter_key: str,
        adapter_label: str,
        adapter_path: str,
        base_model_id: str,
        device: str,
        dtype: str,
        quantization: str,
        load_adapter: bool,
        local_files_only: bool,
        credential_fingerprint: str,
        _token: str,
    ) -> Any:
        del credential_fingerprint
        from attention_maps.inference.local_comparison import (
            LocalAdapterSpec,
            load_local_model_pair,
        )

        adapter_spec = LocalAdapterSpec(
            key=adapter_key,
            label=adapter_label,
            path=Path(adapter_path),
            base_model_id=base_model_id,
        )
        return load_local_model_pair(
            adapter_spec,
            device=device,
            dtype=dtype,
            quantization=quantization,
            load_adapter=load_adapter,
            local_files_only=local_files_only,
            token=_token or None,
        )

    @st.cache_resource(show_spinner=False)
    def cached_arkios(
        model_id: str,
        revision: str,
        device: str,
        dtype: str,
        local_files_only: bool,
    ) -> Any:
        from attention_maps.inference.arkios import load_arkios

        return load_arkios(
            model_id=model_id,
            revision=revision,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
        )

    @st.cache_resource(show_spinner=False)
    def cached_himalayagpt(
        model_id: str,
        revision: str,
        device: str,
        dtype: str,
        local_files_only: bool,
    ) -> Any:
        from attention_maps.inference.himalayagpt import load_himalayagpt

        return load_himalayagpt(
            model_id=model_id,
            revision=revision,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
        )

    @st.cache_resource(show_spinner=False)
    def cached_iriis_gpt2(
        model_key: str,
        device: str,
        dtype: str,
        local_files_only: bool,
        credential_fingerprint: str,
        _token: str,
    ) -> Any:
        del credential_fingerprint
        from attention_maps.inference.iriis_gpt2 import (
            iriis_gpt2_spec,
            load_iriis_gpt2,
        )

        return load_iriis_gpt2(
            iriis_gpt2_spec(model_key),
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            token=_token or None,
        )

    @st.cache_resource(show_spinner=False)
    def cached_gemma4_base(
        model_id: str,
        revision: str,
        device: str,
        dtype: str,
        local_files_only: bool,
        credential_fingerprint: str,
        _token: str,
    ) -> Any:
        del credential_fingerprint
        from attention_maps.inference.gemma4_base import load_gemma4_base

        return load_gemma4_base(
            model_id=model_id,
            revision=revision,
            device=device,
            dtype=dtype,
            local_files_only=local_files_only,
            token=_token or None,
        )

    return {
        "arkios": cached_arkios,
        "flores_examples": cached_flores_examples,
        "gemma4_base": cached_gemma4_base,
        "google_backend": cached_google_backend,
        "himalayagpt": cached_himalayagpt,
        "huggingface_backend": cached_huggingface_backend,
        "iriis_gpt2": cached_iriis_gpt2,
        "lima_teacher": cached_lima_teacher,
        "local_model_pair": cached_local_model_pair,
        "nlue_examples": cached_nlue_examples,
    }
