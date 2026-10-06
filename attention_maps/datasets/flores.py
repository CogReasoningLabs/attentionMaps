"""Authenticated, revision-pinned access to the official FLORES Parquet data."""

from __future__ import annotations

import re
from typing import Any, Sequence


FACEBOOK_FLORES_DATASET_ID = "facebook/flores"
FLORES_ACCESS_HELP = (
    "facebook/flores requires Hugging Face dataset access. Open "
    "https://huggingface.co/datasets/facebook/flores, review and accept its "
    "access conditions, then set HF_TOKEN to a read token from that authorized "
    "account (including gated-repository read permission)."
)


def load_facebook_flores(
    config: str | None,
    *,
    split: str,
    revision: str | None,
    token: str | None,
    filters: Sequence[tuple[str, str, Any]] = (),
) -> tuple[Any, str, int | None]:
    """Validate evaluation splits, check access, and stream the current export.

    Resolving main to a commit avoids stale cached legacy loading scripts.
    An explicit older revision is never replaced by main or another repository.
    Only file metadata is read for sizes; row counts come from the dataset card.
    """

    if split not in {"dev", "devtest"}:
        raise ValueError(
            "facebook/flores has no train split. Choose 'dev' or 'devtest' "
            "and configuration 'npi_Deva' for Nepali (or 'eng_Latn-npi_Deva' "
            "for aligned English/Nepali). FLORES is an evaluation benchmark."
        )
    if not config or (config != "all" and not re.fullmatch(
        r"[a-z]{3}_[A-Z][a-z]{3}(?:-[a-z]{3}_[A-Z][a-z]{3})?", config
    )):
        raise ValueError(
            "Select a FLORES configuration: 'npi_Deva' for Nepali, "
            "'eng_Latn-npi_Deva' for English/Nepali, or 'all' for all languages. "
            "Use split 'dev' or 'devtest'."
        )

    from datasets import load_dataset
    from huggingface_hub import HfApi, get_token
    from huggingface_hub.errors import HfHubHTTPError

    effective_token = token or get_token()
    api = HfApi()
    try:
        api.auth_check(FACEBOOK_FLORES_DATASET_ID, repo_type="dataset", token=effective_token)
        info = api.dataset_info(
            FACEBOOK_FLORES_DATASET_ID, revision=revision or "main",
            token=effective_token, files_metadata=True,
        )
        if not info.sha:
            raise ValueError("Could not resolve the FLORES source revision")
        folder = "all" if config == "all" else (
            f"pair/{config}" if "-" in config else f"language/{config}"
        )
        prefix = f"data/{folder}/{split}-"
        files = [item for item in info.siblings or [] if (
            item.rfilename.startswith(prefix) and item.rfilename.endswith(".parquet")
        )]
        if not files:
            raise ValueError(
                f"No Parquet export for FLORES {config!r}/{split} at this revision. "
                "Check the configuration; for an old script-only revision, use "
                "a script-free export or select the current main revision."
            )
        options = {"filters": list(filters)} if filters else {}
        stream = load_dataset(
            FACEBOOK_FLORES_DATASET_ID, config, split=split, streaming=True,
            revision=info.sha, token=effective_token, **options,
        )
    except HfHubHTTPError as error:
        if getattr(error.response, "status_code", None) in {401, 403}:
            raise ValueError(FLORES_ACCESS_HELP) from error
        raise
    file_bytes = (
        sum(item.size for item in files) if all(item.size is not None for item in files)
        else None
    )
    return stream, info.sha, file_bytes
