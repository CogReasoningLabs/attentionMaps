"""Explicit, reusable parsers for schema-mapped text fields."""

TEXT_FIELD_PARSERS = ("string", "join_strings")
PARSABLE_FIELDS = {"text", "prompt", "chosen", "rejected", "input", "reference"}


def validate_field_parsers(parsers):
    if parsers is None:
        return {}
    if not isinstance(parsers, dict) or any(
        field not in PARSABLE_FIELDS or mode not in TEXT_FIELD_PARSERS
        for field, mode in parsers.items()
    ):
        raise ValueError(
            "field_parsers must map text/prompt/chosen/rejected/input/reference to string or join_strings"
        )
    return dict(parsers)


def parse_field(value, mode, *, field, source):
    """Convert only declared shapes; never stringify arbitrary metadata objects."""
    if value is None:
        return None
    if mode == "string":
        if not isinstance(value, str):
            raise ValueError(f"{field} from {source!r} requires a string; got {type(value).__name__}")
        return value
    if mode == "join_strings":
        if isinstance(value, str):
            return value  # Some sources mix scalar and paragraph-list rows.
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{field} from {source!r} requires a string or list of strings; got {type(value).__name__}")
        for index, item in enumerate(value):
            if not isinstance(item, str):
                raise ValueError(f"{field} from {source!r} has {type(item).__name__} at list item {index}; expected string")
        return "\n\n".join(item for item in value if item.strip())
    raise ValueError(f"Unknown field parser: {mode!r}")
