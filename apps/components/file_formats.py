"""Display file-format metadata calculated by the shared inspector."""

from __future__ import annotations

from typing import Any

_SUPPORT_LABELS = {
    "supported_extension": "Supported extension",
    "decompress_first": "Decompress first",
    "inspect_archive_members": "Inspect archive members",
    "reader_not_implemented": "Reader not implemented",
    "unknown_format": "Unknown format",
}


def render_file_formats(st: Any, metadata: dict | None) -> None:
    st.markdown("#### Source file formats")
    if metadata is None:
        st.caption("File-format details were not saved in this report. Rerun the inspection script to add them.")
        return
    cards = st.columns(2)
    cards[0].metric("File format", metadata["format"].upper())
    cards[1].metric("File extensions", ", ".join(value or "(none)" for value in metadata["extensions"]) or "Unknown")
    if metadata["mixed_formats"]:
        st.info("This selection contains multiple file formats. Choose a reader for each format group.")
    st.caption(metadata["note"])
    with st.expander("Formats and batch parsing requirements", expanded=True):
        st.dataframe([
            {"Extension": group["extension"] or "(none)", "Format": group["format"].upper(),
             "Files": group["files"], "Bytes": group["bytes"],
             "Compression": " → ".join(group["compression"]) or "No outer wrapper",
             "Archive": group["container"] or "—",
             "Local batch support": _SUPPORT_LABELS[group["batch_loader_support"]],
             "Parsing requirement": group["parsing_hint"]}
            for group in metadata["groups"]
        ], hide_index=True, width="stretch")
    with st.expander("Per-file format details"):
        st.dataframe([
            {"File": item["path"], "Extension": item["extension"] or "(none)",
             "Format": item["format"].upper(), "Bytes": item["bytes"],
             "Compression": " → ".join(item["compression"]) or "No outer wrapper",
             "Archive": item["container"] or "—",
             "Batch reader": item["batch_reader"] or "Undetermined",
             "Local batch support": _SUPPORT_LABELS[item["batch_loader_support"]]}
            for item in metadata["files"]
        ], hide_index=True, width="stretch")
