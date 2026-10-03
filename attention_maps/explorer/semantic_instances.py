"""Schema-aware atomic instances and deterministic semantic representations."""

import hashlib
import json
from pathlib import Path

from attention_maps.datasets.schemas import STANDARD_DATASET_SCHEMAS
from .inspection_runs import iter_inspection_records
from .field_parsers import parse_field, validate_field_parsers

SCHEMAS = {schema.key: schema for schema in STANDARD_DATASET_SCHEMAS}
FIELD_NAMES = {field for schema in SCHEMAS.values() for field in schema.required_fields}
UNITS = {"pretraining": "one document", "instruction_finetuning": "one complete conversation",
         "task_specific_supervised": "one labelled example", "preference_tuning": "one prompt/chosen/rejected group",
         "evaluation": "one input/reference example"}


def path_value(record, path):
    value = record
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def display_value(value):
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def resolve_instance_definition(inventory, fields, args):
    schema = getattr(args, "training_schema", "auto")
    mapping = getattr(args, "field_mapping", None) or {}
    parsers = validate_field_parsers(getattr(args, "field_parsers", None))
    if not isinstance(mapping, dict):
        raise ValueError("field_mapping must be a mapping of canonical fields to source paths")
    available = set(inventory["columns"]) | set(mapping)
    if schema == "auto":
        if {"prompt", "chosen", "rejected"} <= available:
            schema = "preference_tuning"
        elif available & {"messages", "conversations"} or ("instruction" in available and available & {"output", "response", "answer"}):
            schema = "instruction_finetuning"
        elif available & {"label", "labels"}:
            schema = "task_specific_supervised"
        else:
            schema = "pretraining"
    if schema not in SCHEMAS:
        raise ValueError("Choose a supported dataset instance schema")
    for target, source in mapping.items():
        if target not in FIELD_NAMES or not isinstance(source, str) or not source.strip():
            raise ValueError("field_mapping must map canonical field names to non-empty source paths")
        if source.split(".")[0] not in inventory["columns"]:
            raise ValueError(f"Mapped field {target}: source path {source!r} is not present")
    if schema in {"pretraining", "task_specific_supervised"}:
        if "text" in mapping:
            fields = [mapping["text"]]
        elif schema == "pretraining" and not getattr(args, "text_columns", None):
            # An unconfigured document is one natural-language field, not every
            # string-valued metadata column or a list of paragraph strings.
            types = {field["column"]: field["type"].lower() for field in inventory.get("schema", [])}
            fields = [field for field in fields if "string" in types.get(field, "string")
                      and not any(part in types.get(field, "").lower() for part in ("list", "sequence", "struct", "map"))][:1]
        if not fields or any(field.split(".")[0] not in inventory["columns"] for field in fields):
            raise ValueError("Choose valid text_columns or field_mapping.text for this schema")
    else:
        fields = []  # Structured instances use their complete mapped fields.
    unit = getattr(args, "text_record_unit", "line")
    if unit not in {"line", "blank_line"}:
        raise ValueError("text_record_unit must be line or blank_line")
    if unit != "line" and inventory["format"] != "text":
        raise ValueError("text_record_unit: blank_line applies only to a local/staged TXT source")
    return {"version": 1, "schema": schema, "label": SCHEMAS[schema].label, "unit": UNITS[schema],
            "required_fields": list(SCHEMAS[schema].required_fields), "field_mapping": mapping, "field_parsers": parsers,
            "text_columns": list(fields), "text_record_unit": unit if inventory["format"] == "text" else "source_row",
            "task_name": getattr(args, "task_name", None),
            "embedding_policy": {
                "pretraining": "Full document text",
                "instruction_finetuning": "All messages in order, with role markers",
                "task_specific_supervised": "Task, complete input text, and label, with field markers",
                "preference_tuning": "Prompt, chosen, and rejected together, with field/role markers",
                "evaluation": "Input and reference together, with field markers",
            }[schema],
            "metadata_policy": "IDs, split and provenance are displayed and retained, but excluded from the embedding text. Unknown splits remain null; this adapter does not assign training splits."}


def _messages(value):
    if not isinstance(value, list) or not value:
        raise ValueError("messages must contain a complete ordered conversation")
    result = []
    previous = None
    for message in value:
        if not isinstance(message, dict):
            raise ValueError("Each message must have a role and content")
        role = message.get("role", message.get("from"))
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        content = message.get("content", message.get("value", message.get("text")))
        if role not in {"system", "user", "assistant"}:
            raise ValueError("Each message needs a system/user/assistant role")
        if not display_value(content).strip():
            raise ValueError(f"{role} message has empty content")
        if (role == "system" and result) or (role == "assistant" and previous != "user") or (role == "user" and previous == "user"):
            raise ValueError("Invalid conversation role order; keep complete user/assistant turns together")
        result.append({**message, "role": role, "content": content})
        previous = role
    if result[-1]["role"] != "assistant":
        raise ValueError("Instruction instances must end with an assistant response")
    return result


def conversation_text(messages):
    return "\n\n".join(f"[{message['role']}]\n{display_value(message['content'])}" for message in messages)


def branch_text(value):
    # Preference branches can themselves contain ordered chat messages.
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, dict) and item.get("role") in {"system", "user", "assistant"}
                                       and display_value(item.get("content")).strip() for item in value):
        return conversation_text(value)
    raise ValueError("Preference fields must be text or ordered messages with role and content")


def make_instance(record, definition, *, source_row, source_identity, source_split=None, hash_generated_id=True):
    mapping = definition["field_mapping"]
    parsers = definition.get("field_parsers") or {}

    def field(name, *aliases):
        if name in mapping:
            source = mapping[name]
            value = path_value(record, source)
        else:
            source = next((key for key in (name, *aliases) if key in record), name)
            value = record.get(source)
        return parse_field(value, parsers[name], field=name, source=source) if name in parsers else value

    schema = definition["schema"]
    # The raw record remains separately intact; canonical fields are never written back to it.
    instance = {key: record[key] for key in ("source", "domain", "language", "license", "metadata") if key in record}
    identifier = field("id")
    generated = identifier is None or str(identifier).strip() == ""
    if generated:
        if hash_generated_id:
            digest = hashlib.sha256(json.dumps([source_identity, source_row, record], ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()
            identifier = "source-" + digest[:24]
        else:
            identifier = f"inspection-row-{source_row}"
    if type(identifier) not in (str, int):
        raise ValueError("Instance id must be a string or integer")
    split = field("split")
    # A source split can be a language/domain partition (for example, "nep"),
    # which is not a train/validation/test assignment. Preserve its name
    # separately without inventing a canonical training split.
    if source_split is not None:
        instance["source_split"] = source_split
    if split is None and source_split in {"train", "validation", "test", "val", "dev"}:
        split = source_split
    if split == "val" or split == "dev":
        split = "validation"
    if split is not None and split not in ("train", "validation", "test"):
        raise ValueError("Instance split must be train, validation, test, or unknown (null)")
    instance.update(id=identifier, split=split)
    if schema in {"pretraining", "task_specific_supervised"}:
        parts = [path_value(record, path) for path in definition["text_columns"]]
        if "text" in parsers:
            parts = [parse_field(value, parsers["text"], field="text", source=path)
                     for path, value in zip(definition["text_columns"], parts)]
        invalid = [(path, type(value).__name__) for path, value in zip(definition["text_columns"], parts)
                   if value is not None and not isinstance(value, str)]
        if invalid:
            raise ValueError(f"Document/example text fields must be strings; invalid fields: {invalid}. "
                             "Map text fields rather than metadata objects")
        text = "\n\n".join(value for value in parts if value is not None and value.strip())
        instance["text"] = text
        if schema == "task_specific_supervised":
            label = field("label", "labels")
            task = field("task") or definition["task_name"]
            if label is None or not display_value(label).strip() or not isinstance(task, str) or not task.strip():
                raise ValueError("Supervised instances require label and task (or set task_name in the embedding config)")
            instance.update(label=label, task=task)
            text = f"[task]\n{task}\n\n[text]\n{text}\n\n[label]\n{display_value(label)}" if text.strip() else ""
    elif schema == "evaluation":
        for name in ("input", "reference"):
            value = field(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Evaluation instances require a non-empty string {name}")
            instance[name] = value
        task = field("task") or definition["task_name"]
        if task is not None and (not isinstance(task, str) or not task.strip()):
            raise ValueError("Evaluation task must be non-empty text when provided")
        if task is not None:
            instance["task"] = task
        text = f"[input]\n{instance['input']}\n\n[reference]\n{instance['reference']}"
    elif schema == "instruction_finetuning":
        messages = field("messages", "conversations")
        if messages is None and "instruction" in record:
            messages = []
            if record.get("system"):
                messages.append({"role": "system", "content": record["system"]})
            instruction = record["instruction"]
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("instruction must be non-empty text")
            extra = record.get("input")
            if extra:
                instruction += "\n\n[input]\n" + display_value(extra)
            messages.append({"role": "user", "content": instruction})
            output = next((record[key] for key in ("output", "response", "answer") if key in record), None)
            messages.append({"role": "assistant", "content": output})
        instance["messages"] = _messages(messages)
        text = conversation_text(instance["messages"])
    else:
        for name in ("prompt", "chosen", "rejected"):
            value = field(name)
            if value is None or not branch_text(value).strip():
                raise ValueError(f"Preference instances require a non-empty {name} field")
            instance[name] = value
        if " ".join(branch_text(instance["chosen"]).split()) == " ".join(branch_text(instance["rejected"]).split()):
            raise ValueError("Preference chosen and rejected responses must differ")
        text = "\n\n".join(f"[{name}]\n{branch_text(instance[name])}" for name in ("prompt", "chosen", "rejected"))
    return instance, text, generated


def inspection_instance_text(record, definition, *, source_row=0, source_identity="inspection", source_split=None):
    """Validate an atomic instance and return its language-bearing content."""
    instance, _, _ = make_instance(
        record, definition, source_row=source_row, source_identity=source_identity,
        source_split=source_split, hash_generated_id=False,
    )
    schema = definition["schema"]
    if schema in {"pretraining", "task_specific_supervised"}:
        return instance["text"]
    if schema == "evaluation":
        return "\n\n".join((instance["input"], instance["reference"]))
    if schema == "instruction_finetuning":
        mapped_messages = definition["field_mapping"].get("messages")
        explicit_messages = (path_value(record, mapped_messages) if mapped_messages
                             else record.get("messages", record.get("conversations")))
        if explicit_messages is None and "instruction" in record:
            # make_instance adds a synthetic [input] marker for embedding, but
            # script/language evidence must only count source-authored content.
            answer = next((record[key] for key in ("output", "response", "answer") if key in record), None)
            return "\n\n".join(display_value(value) for value in
                                 (record.get("system"), record["instruction"], record.get("input"), answer)
                                 if display_value(value).strip())
        return "\n\n".join(display_value(message["content"]) for message in instance["messages"])
    def content(value):
        return value if isinstance(value, str) else "\n\n".join(display_value(message["content"]) for message in value)
    return "\n\n".join(content(instance[name]) for name in ("prompt", "chosen", "rejected"))


def iter_instance_records(inventory, definition, *, token=None, batch_size=1024):
    if definition["text_record_unit"] != "blank_line":
        yield from iter_inspection_records(inventory, token=token, batch_size=batch_size)
        return
    if inventory.get("row_filters"):
        raise ValueError("blank_line text records do not have language metadata; choose a language-specific file")
    # Explicit blank-line document boundaries; preserve all internal lines.
    parts = []
    with Path(inventory["path"]).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if line.strip():
                parts.append(line)
            elif parts:
                yield {"text": "".join(parts).rstrip("\r\n")}
                parts = []
        if parts:
            yield {"text": "".join(parts).rstrip("\r\n")}
