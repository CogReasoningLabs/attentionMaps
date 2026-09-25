"""Exact metadata selection shared by inspection and embedding readers."""

from collections.abc import Mapping


def validate_row_filters(filters):
    if filters is None:
        return
    if not isinstance(filters, dict):
        raise ValueError("row_filters must map column paths to non-empty lists of exact string values")
    for column, values in filters.items():
        if (not isinstance(column, str) or not column.strip()
                or any(not part.strip() for part in column.split('.'))
                or not isinstance(values, list) or not values
                or any(not isinstance(value, str) or not value.strip() for value in values)):
            raise ValueError("row_filters must map column paths to non-empty lists of exact string values")


def parse_row_filters(items):
    filters = {}
    for item in items or ():
        column, separator, value = item.partition('=')
        if not separator:
            raise ValueError("--row-filter requires COLUMN=VALUE; repeat for alternatives or additional columns")
        filters.setdefault(column, []).append(value)
    validate_row_filters(filters)
    return filters


def field_value(record, column):
    # Literal column names take priority over nested paths.
    if column in record:
        return record[column]
    value = record
    for part in column.split('.'):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def matches_row_filters(record, filters):
    """AND between columns, OR between values; list-valued labels use membership."""
    for column, allowed in (filters or {}).items():
        value = field_value(record, column)
        values = value if isinstance(value, (list, tuple)) else (value,)
        if not any(isinstance(item, str) and item in allowed for item in values):
            return False
    return True


def with_row_filters(inventory, filters):
    validate_row_filters(filters)
    if not filters:
        return inventory
    unknown = [column for column in filters
               if column not in inventory['columns'] and column.split('.')[0] not in inventory['columns']]
    if unknown:
        raise ValueError(f"Unknown row-filter columns: {unknown}. Available columns: {inventory['columns']}")
    if inventory.get('format') == 'text':
        raise ValueError("Plain text has no language metadata columns; choose a language-specific file instead")
    return {**inventory, 'row_filters': filters,
            'rows_before_selection': inventory.get('rows'), 'rows': None}
