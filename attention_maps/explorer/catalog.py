"""Dataset catalog, taxonomy, and local/remote source discovery."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from attention_maps.eda.text import load_stopwords_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_FINETUNED_MODELS_ROOT = PROJECT_ROOT / "finetuned_models"
DEFAULT_NEPALI_STOPWORDS_PATH = PROJECT_ROOT / "configs" / "eda" / "stopwords.txt"
EDA_CLEANING_NOTES_PATH = PROJECT_ROOT / "docs" / "eda-cleaning-notes.md"
ARKIOS_BACKEND_NAME = "Arkios 1B Chat · local"
HIMALAYAGPT_BACKEND_NAME = "HimalayaGPT 0.5B Instruct · local"
GEMMA4_BASE_BACKEND_NAME = "Gemma 4 E2B Base · Hugging Face local"
GOOGLE_GEMMA_BACKEND_NAME = "Gemma 4 · Google API"
DEFAULT_LIMA_TRANSLATIONS_PATH = (
    PROJECT_ROOT / "data_generation_pipeline" / "lima_translations.json"
)
HIMALAYA_NEPALI_SFT_DATASET_ID = "himalaya-ai/nepali-sft-dataset"
AYA_DATASET_ID = "CohereLabs/aya_dataset"
AYA_NEPALI_LANGUAGE_CODE = "npi"
IRIIS_NEPALI_TEXT_CORPUS_ID = "IRIIS-RESEARCH/Nepali-Text-Corpus"
KAGGLE_MOVIE_REVIEWS_DATASET_ID = (
    "shikharghimire/nepali-language-sentiment-analysis-movie-reviews"
)
KAGGLE_HATE_SPEECH_DATASET_ID = "mohanbhandari/nepali-hate-speech-collection"
KAGGLE_OSCAR_NEPALI_DATASET_ID = "hsebarp/oscar-corpus-nepali"
KAGGLE_OSCAR_DEDUP_FILE = "ne_dedup.txt"
KAGGLE_OSCAR_DEDUP_APPROX_BYTES = 1_240_000_000
DEFAULT_EDA_TERMS = (
    "मन्त्रालय",
    "लिलाम",
    "सरकार",
    "कार्यालय",
    "विभाग",
    "समिति",
    "आयोग",
    "पालिका",
)
ALL_PROVIDERS = "All providers"
ALL_PURPOSES = "All purposes"
LOCAL_PROVIDER = "Local project"
HIMALAYA_AI_PROVIDER = "Himalaya AI"
ARKIOS_PROVIDER = "Arkios"
GOOGLE_PROVIDER = "Google (Gemini / Gemma)"
COHERE_PROVIDER = "CohereLabs"
IRIIS_PROVIDER = "IRIIS Research"
GAIR_PROVIDER = "GAIR / LIMA"
KAGGLE_COMMUNITY_PROVIDER = "Kaggle community"
UNCLASSIFIED_PROVIDER = "Provider pending review"
PURPOSE_ROOTS = (
    "Pretraining corpus",
    "Instruction fine-tuning",
    "Preference tuning",
    "Task-specific fine-tuning",
    "Tokenizer development",
    "Evaluation / benchmark",
    "Unclassified — pending review",
)
UNCLASSIFIED_PURPOSE = ("Unclassified — pending review",)
TRACKED_PROVIDERS = (
    HIMALAYA_AI_PROVIDER,
    ARKIOS_PROVIDER,
    GOOGLE_PROVIDER,
    LOCAL_PROVIDER,
)
HIMALAYA_DATASET_CATALOG = (
    (
        "himalaya-ai/nepali-corpus-compile",
        "Nepali Corpus Compile",
        ("Pretraining corpus", "Aggregated corpus"),
        ("Nepali", "compiled"),
    ),
    (
        "himalaya-ai/sangraha_nepali",
        "Sangraha Nepali",
        ("Pretraining corpus", "General corpus"),
        ("Nepali",),
    ),
    (
        "himalaya-ai/nepali-news-corpus",
        "Nepali News Corpus",
        ("Pretraining corpus", "News domain"),
        ("Nepali", "news"),
    ),
    (
        "himalaya-ai/cc100-nepali",
        "CC100 Nepali",
        ("Pretraining corpus", "Web corpus"),
        ("Nepali", "web"),
    ),
    (
        "himalaya-ai/gpt2-pretrain-corpus",
        "GPT-2 Pretraining Corpus",
        ("Pretraining corpus", "Model-oriented corpus"),
        ("Nepali", "GPT-2"),
    ),
    (
        "himalaya-ai/nepali-roman-pretrain",
        "Nepali Romanized Pretraining",
        ("Pretraining corpus", "Romanized corpus"),
        ("Nepali", "Romanized"),
    ),
    (
        "himalaya-ai/nepali-pretrain-corpus",
        "Nepali Pretraining Corpus",
        ("Pretraining corpus", "General corpus"),
        ("Nepali",),
    ),
    (
        "himalaya-ai/nepali-tokenizer-corpus",
        "Nepali Tokenizer Corpus",
        ("Tokenizer development", "Tokenizer-training corpus"),
        ("Nepali", "tokenizer"),
    ),
    (
        "himalaya-ai/nepali-sft-compile",
        "Nepali SFT Compile",
        ("Instruction fine-tuning", "Aggregated instruction corpus"),
        ("Nepali", "SFT", "compiled"),
    ),
    (
        HIMALAYA_NEPALI_SFT_DATASET_ID,
        "Nepali SFT Dataset",
        ("Instruction fine-tuning", "General instruction following"),
        ("Nepali", "SFT"),
    ),
    (
        "himalaya-ai/nepali-proofreader",
        "Nepali Proofreader",
        ("Task-specific fine-tuning", "Proofreading"),
        ("Nepali", "proofreading"),
    ),
    (
        "himalaya-ai/deva-sft-compile-v1",
        "Devanagari SFT Compile v1",
        ("Instruction fine-tuning", "Aggregated instruction corpus"),
        ("Devanagari", "SFT", "compiled"),
    ),
    (
        "himalaya-ai/nepali_pdf_corpus",
        "Nepali PDF Corpus",
        ("Pretraining corpus", "Document / PDF corpus"),
        ("Nepali", "PDF"),
    ),
    (
        "himalaya-ai/nepali-hermes-function-calling-v1",
        "Nepali Hermes Function Calling v1",
        ("Instruction fine-tuning", "Function calling"),
        ("Nepali", "tools", "structured output"),
    ),
    (
        "himalaya-ai/nepali-json-mode-singleturn",
        "Nepali JSON Mode Single-turn",
        ("Instruction fine-tuning", "Structured output"),
        ("Nepali", "JSON", "single-turn"),
    ),
    (
        "himalaya-ai/nepali-honorific-bench",
        "Nepali Honorific Benchmark",
        ("Evaluation / benchmark", "Honorific language"),
        ("Nepali", "benchmark"),
    ),
    (
        "himalaya-ai/sft-harl-full-v1",
        "SFT HARL Full v1",
        ("Instruction fine-tuning", "Subtype pending review"),
        ("Nepali", "SFT", "needs taxonomy review"),
    ),
)


def configured_finetuned_models_root() -> Path:
    """Resolve an optional shared model directory for worktree-based runs."""

    configured = os.getenv("ATTENTION_MAPS_FINETUNED_MODELS_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_FINETUNED_MODELS_ROOT


def configured_eda_output_root() -> Path:
    """Resolve the derived-artifact directory used by Streamlit EDA runs."""

    configured = os.getenv("ATTENTION_MAPS_EDA_OUTPUT_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return PROJECT_ROOT / "artifacts" / "eda" / "ui"


def configured_nepali_stopwords() -> tuple[str, ...]:
    """Load the repository's canonical Nepali stopword resource."""

    if DEFAULT_NEPALI_STOPWORDS_PATH.is_file():
        return load_stopwords_file(DEFAULT_NEPALI_STOPWORDS_PATH)
    return DEFAULT_WORDCLOUD_STOPWORDS


def load_eda_cleaning_notes(path: Path = EDA_CLEANING_NOTES_PATH) -> str:
    """Load the version-controlled EDA and cleaning methodology for the UI."""

    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        return f"# EDA and cleaning notes\n\nCould not load `{path}`: {error}"


SPLIT_NAMES = ("train", "validation", "test")
PIPELINE_STAGES = ("raw", "cleaned", "processed", "tokenized")
PIPELINE_STAGE_LABELS = {
    "raw": "Raw",
    "cleaned": "Cleaned",
    "processed": "Preprocessed",
    "tokenized": "Tokenized",
}
MANIFEST_NAMES = (
    "download_manifest.json",
    "cleaning_manifest.json",
    "build_manifest.json",
    "tokenization_manifest.json",
    "dataset_info.json",
)
DATASET_LABELS = {
    "nepcov19tweets": "NepCOV19Tweets · sentiment",
}
SENTIMENT_LABELS = {-1: "negative", 0: "neutral", 1: "positive"}
VIEWER_PREFIX = "__viewer_"
TEXT_FIELD_NAMES = (
    "source_text",
    "translation",
    "text",
    "content",
    "article",
    "Article",
    "lyrics",
    "Lyrics",
    "messages",
    "conversations",
    "inputs",
    "targets",
    "prompt",
    "completion",
    "response",
    "chosen",
    "rejected",
)
DEFAULT_WORDCLOUD_STOPWORDS = (
    "अनि",
    "अब",
    "अथवा",
    "अरू",
    "एक",
    "एउटा",
    "का",
    "कि",
    "की",
    "को",
    "छ",
    "छन्",
    "तथा",
    "तर",
    "त्यो",
    "नै",
    "पनि",
    "भएको",
    "भने",
    "सम्बन्धी",
    "सम्बन्धित",
    "मा",
    "र",
    "लाई",
    "लागि",
    "ले",
    "वा",
    "यो",
    "हो",
)
ENGLISH_WORDCLOUD_STOPWORDS = (
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
)
DEVANAGARI_FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSerifDevanagari-Regular.ttf"),
    Path("/usr/share/fonts/truetype/lohit-nepali/Lohit-Nepali.ttf"),
    Path("/usr/share/fonts/truetype/msttcorefonts/Nirmala.ttf"),
    Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
)
LATIN_FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
)


@dataclass(frozen=True)
class DatasetSpec:
    """A logical dataset represented by one or more data files."""

    key: str
    label: str
    stage: str
    files: tuple[Path, ...]
    format: str = "parquet"
    visible_columns: tuple[str, ...] | None = None
    dataset_id: str | None = None
    dataset_config: str | None = None
    dataset_split: str | None = None
    dataset_file: str | None = None
    filter_column: str | None = None
    filter_value: str | None = None
    download_bytes: int | None = None
    provider: str = UNCLASSIFIED_PROVIDER
    purpose_path: tuple[str, ...] = UNCLASSIFIED_PURPOSE
    tags: tuple[str, ...] = ()
    source_provider: str | None = None
    adapted_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("dataset provider cannot be empty")
        if not self.purpose_path or any(not item.strip() for item in self.purpose_path):
            raise ValueError("dataset purpose hierarchy cannot be empty")

    @property
    def primary_purpose(self) -> str:
        return self.purpose_path[0]

    @property
    def purpose_label(self) -> str:
        return " › ".join(self.purpose_path)

    @property
    def provider_facets(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                provider
                for provider in (self.provider, self.source_provider, *self.adapted_by)
                if provider
            )
        )

    @property
    def location(self) -> Path | str:
        if self.format.startswith("kaggle") and self.dataset_id and self.dataset_file:
            return f"kaggle://datasets/{self.dataset_id}/{self.dataset_file}"
        if self.dataset_id:
            config = f"/{self.dataset_config}" if self.dataset_config else ""
            split = f"/{self.dataset_split}" if self.dataset_split else ""
            location = f"hf://datasets/{self.dataset_id}{config}{split}"
            if self.filter_column and self.filter_value:
                location += f"?{self.filter_column}={self.filter_value}"
            return location
        if len(self.files) == 1:
            return self.files[0]
        return Path(_common_path(self.files))


def filter_dataset_specs(
    specs: Sequence[DatasetSpec],
    *,
    provider: str = ALL_PROVIDERS,
    purpose: str = ALL_PURPOSES,
    stage: str | None = None,
) -> list[DatasetSpec]:
    """Filter catalog entries across independent provider and purpose facets."""

    return [
        spec
        for spec in specs
        if (provider == ALL_PROVIDERS or provider in spec.provider_facets)
        and (purpose == ALL_PURPOSES or purpose == spec.primary_purpose)
        and (stage is None or spec.stage == "dataset" or spec.stage == stage)
    ]


def available_dataset_purposes(
    specs: Sequence[DatasetSpec],
    *,
    provider: str = ALL_PROVIDERS,
    stage: str | None = None,
) -> tuple[str, ...]:
    """Return only human-assigned purposes available for the active facets."""

    available = {
        spec.primary_purpose
        for spec in filter_dataset_specs(
            specs,
            provider=provider,
            stage=stage,
        )
    }
    ordered = [purpose for purpose in PURPOSE_ROOTS if purpose in available]
    ordered.extend(sorted(available.difference(PURPOSE_ROOTS)))
    return (ALL_PURPOSES, *ordered)


def _common_path(paths: Sequence[Path]) -> str:
    """Return a common path without importing platform-specific helpers."""

    import os

    return os.path.commonpath([str(path) for path in paths])


def _parquet_files(path: Path, *, recursive: bool = True) -> tuple[Path, ...]:
    if not path.exists():
        return ()
    if path.is_file():
        return (path.resolve(),) if path.suffix.lower() == ".parquet" else ()
    iterator = path.rglob("*.parquet") if recursive else path.glob("*.parquet")
    return tuple(sorted(item.resolve() for item in iterator if item.is_file()))


def _add_spec(
    specs: list[DatasetSpec],
    seen: set[tuple[str, ...]],
    *,
    key: str,
    label: str,
    stage: str,
    files: Iterable[Path],
) -> None:
    resolved = tuple(sorted(Path(path).resolve() for path in files))
    signature = tuple(str(path) for path in resolved)
    if not resolved or signature in seen:
        return
    seen.add(signature)
    specs.append(
        DatasetSpec(
            key=key,
            label=label,
            stage=stage,
            files=resolved,
            provider=LOCAL_PROVIDER,
            tags=(stage, "purpose pending review"),
        )
    )


def discover_datasets(data_root: Path) -> list[DatasetSpec]:
    """Discover the four supported pipeline stages under ``data_root``."""

    root = data_root.expanduser().resolve()
    specs: list[DatasetSpec] = []
    seen: set[tuple[str, ...]] = set()

    raw_root = root / "raw"
    if raw_root.is_dir():
        for source_dir in sorted(path for path in raw_root.iterdir() if path.is_dir()):
            files = _parquet_files(source_dir / "data")
            _add_spec(
                specs,
                seen,
                key=f"raw:{source_dir.name}",
                label=DATASET_LABELS.get(source_dir.name, source_dir.name),
                stage="raw",
                files=files,
            )

    cleaned_root = root / "cleaned"
    if cleaned_root.is_dir():
        for source_dir in sorted(
            path for path in cleaned_root.iterdir() if path.is_dir()
        ):
            _add_spec(
                specs,
                seen,
                key=f"cleaned:{source_dir.name}",
                label=source_dir.name,
                stage="cleaned",
                files=_parquet_files(source_dir / "data"),
            )

    processed_root = root / "processed"
    if processed_root.is_dir():
        for dataset_dir in sorted(
            path for path in processed_root.iterdir() if path.is_dir()
        ):
            for split in SPLIT_NAMES:
                split_file = dataset_dir / f"{split}.parquet"
                _add_spec(
                    specs,
                    seen,
                    key=f"processed:{dataset_dir.name}:{split}",
                    label=f"{dataset_dir.name} · {split}",
                    stage="processed",
                    files=_parquet_files(split_file),
                )

    tokenized_root = root / "tokenized"
    if tokenized_root.is_dir():
        for dataset_dir in sorted(
            path for path in tokenized_root.iterdir() if path.is_dir()
        ):
            for split in SPLIT_NAMES:
                _add_spec(
                    specs,
                    seen,
                    key=f"tokenized:{dataset_dir.name}:{split}",
                    label=f"{dataset_dir.name} · {split}",
                    stage="tokenized",
                    files=_parquet_files(dataset_dir / split),
                )

    return sorted(specs, key=lambda spec: (spec.stage, spec.label))


def custom_dataset(path: Path) -> DatasetSpec | None:
    """Create a dataset specification from a custom Parquet file or directory."""

    files = _parquet_files(path.expanduser())
    if not files:
        return None
    return DatasetSpec(
        key=f"custom:{path.expanduser().resolve()}",
        label=path.expanduser().resolve().name,
        stage="custom",
        files=files,
        provider=LOCAL_PROVIDER,
        tags=("custom", "purpose pending review"),
    )


def lima_translation_dataset(
    path: Path = DEFAULT_LIMA_TRANSLATIONS_PATH,
) -> DatasetSpec | None:
    """Return the translated-Nepali view of the repository's LIMA JSON."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return None
    return DatasetSpec(
        key=f"lima:translated:{resolved}",
        label="LIMA · Translated Nepali (Gemini/Gemma)",
        stage="dataset",
        files=(resolved,),
        format="json",
        visible_columns=("index", "translation", "status"),
        provider=LOCAL_PROVIDER,
        purpose_path=("Instruction fine-tuning", "Translated instruction corpus"),
        tags=("LIMA", "Nepali", "translated", "synthetic adaptation"),
        source_provider=GAIR_PROVIDER,
        adapted_by=(GOOGLE_PROVIDER,),
    )


def lima_original_dataset(
    path: Path = DEFAULT_LIMA_TRANSLATIONS_PATH,
) -> DatasetSpec | None:
    """Return the original-English view captured before LIMA translation."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return None
    return DatasetSpec(
        key=f"lima:original:{resolved}",
        label="LIMA · Original English",
        stage="dataset",
        files=(resolved,),
        format="json",
        visible_columns=("index", "source_text", "source"),
        provider=GAIR_PROVIDER,
        purpose_path=("Instruction fine-tuning", "General instruction following"),
        tags=("LIMA", "English"),
    )


def lima_synthetic_dataset_specs(
    path: Path = DEFAULT_LIMA_TRANSLATIONS_PATH,
) -> tuple[DatasetSpec, ...]:
    """Return LIMA pipeline artifacts, with generated output shown first."""

    return tuple(
        spec
        for spec in (
            lima_translation_dataset(path),
            lima_original_dataset(path),
        )
        if spec is not None
    )


def himalaya_nepali_sft_dataset() -> DatasetSpec:
    """Return the remote Himalaya Nepali SFT dataset specification."""

    return next(
        spec
        for spec in himalaya_ai_dataset_specs()
        if spec.dataset_id == HIMALAYA_NEPALI_SFT_DATASET_ID
    )


def himalaya_ai_dataset_specs() -> list[DatasetSpec]:
    """Return the purpose-classified Himalaya AI dataset catalog."""

    return [
        DatasetSpec(
            key=f"huggingface:{dataset_id}:train",
            label=f"Himalaya AI · {label}",
            stage="dataset",
            files=(),
            format="huggingface",
            dataset_id=dataset_id,
            dataset_split="train",
            provider=HIMALAYA_AI_PROVIDER,
            purpose_path=purpose_path,
            tags=tags,
        )
        for dataset_id, label, purpose_path, tags in HIMALAYA_DATASET_CATALOG
    ]


def aya_nepali_dataset_specs() -> list[DatasetSpec]:
    """Return the Aya training view filtered to its available Nepali rows."""

    return [
        DatasetSpec(
            key=f"huggingface:{AYA_DATASET_ID}:train:npi",
            label="Cohere Aya · Nepali (train)",
            stage="dataset",
            files=(),
            format="huggingface",
            dataset_id=AYA_DATASET_ID,
            dataset_config="default",
            dataset_split="train",
            filter_column="language_code",
            filter_value=AYA_NEPALI_LANGUAGE_CODE,
            provider=COHERE_PROVIDER,
            purpose_path=(
                "Instruction fine-tuning",
                "Multilingual instruction following",
            ),
            tags=("Nepali", "multilingual", "SFT"),
        )
    ]


def iriis_nepali_text_corpus_specs() -> list[DatasetSpec]:
    """Return streaming train/test views of the IRIIS Nepali text corpus."""

    return [
        DatasetSpec(
            key=f"huggingface:{IRIIS_NEPALI_TEXT_CORPUS_ID}:{split}",
            label=f"IRIIS · Nepali Text Corpus · {split}",
            stage="dataset",
            files=(),
            format="huggingface",
            dataset_id=IRIIS_NEPALI_TEXT_CORPUS_ID,
            dataset_config="default",
            dataset_split=split,
            provider=IRIIS_PROVIDER,
            purpose_path=("Pretraining corpus", "General text corpus"),
            tags=("Nepali", "text corpus"),
        )
        for split in ("train", "test")
    ]


def kaggle_dataset_specs() -> list[DatasetSpec]:
    """Return the public Nepali classification workbooks hosted on Kaggle."""

    return [
        DatasetSpec(
            key=f"kaggle:{KAGGLE_MOVIE_REVIEWS_DATASET_ID}:reviews",
            label="Kaggle · Nepali Movie Reviews (sentiment)",
            stage="dataset",
            files=(),
            format="kaggle",
            dataset_id=KAGGLE_MOVIE_REVIEWS_DATASET_ID,
            dataset_file="nepalimoviereviews.csv.xlsx",
            provider=KAGGLE_COMMUNITY_PROVIDER,
            purpose_path=("Task-specific fine-tuning", "Sentiment classification"),
            tags=("Nepali", "movie reviews", "classification"),
        ),
        DatasetSpec(
            key=f"kaggle:{KAGGLE_HATE_SPEECH_DATASET_ID}:lexicon",
            label="Kaggle · Nepali Hate Speech (lexicon)",
            stage="dataset",
            files=(),
            format="kaggle",
            dataset_id=KAGGLE_HATE_SPEECH_DATASET_ID,
            dataset_file="Nepali hate speech.xlsx",
            provider=KAGGLE_COMMUNITY_PROVIDER,
            purpose_path=("Task-specific fine-tuning", "Hate-speech lexicon"),
            tags=("Nepali", "hate speech", "lexicon"),
        ),
        DatasetSpec(
            key=f"kaggle:{KAGGLE_HATE_SPEECH_DATASET_ID}:tweets",
            label="Kaggle · Nepali Hate Speech (tweets)",
            stage="dataset",
            files=(),
            format="kaggle",
            dataset_id=KAGGLE_HATE_SPEECH_DATASET_ID,
            dataset_file="brb.xlsx",
            provider=KAGGLE_COMMUNITY_PROVIDER,
            purpose_path=("Task-specific fine-tuning", "Hate-speech classification"),
            tags=("Nepali", "hate speech", "tweets", "classification"),
        ),
        DatasetSpec(
            key=f"kaggle:{KAGGLE_OSCAR_NEPALI_DATASET_ID}:deduplicated",
            label="Kaggle · OSCAR Nepali Corpus (deduplicated)",
            stage="dataset",
            files=(),
            format="kaggle_text",
            dataset_id=KAGGLE_OSCAR_NEPALI_DATASET_ID,
            dataset_file=KAGGLE_OSCAR_DEDUP_FILE,
            download_bytes=KAGGLE_OSCAR_DEDUP_APPROX_BYTES,
            provider=KAGGLE_COMMUNITY_PROVIDER,
            purpose_path=("Pretraining corpus", "Web corpus"),
            tags=("Nepali", "OSCAR", "deduplicated", "web"),
        ),
    ]
