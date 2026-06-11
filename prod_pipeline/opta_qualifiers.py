from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


QUALIFIERS_PATH = Path(__file__).resolve().parent.parent / 'config' / 'opta_qualifiers.json'


def _normalize_name(value: Any) -> str:
    text = '' if value is None else str(value)
    return re.sub(r'[^a-z0-9]+', '', text.lower())


@lru_cache(maxsize=1)
def load_opta_qualifier_catalog() -> dict[str, Any]:
    with QUALIFIERS_PATH.open('r', encoding='utf-8') as f:
        data = json.load(f)

    by_id: dict[int, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    categories = data.get('categories', {})
    for category, qualifiers in categories.items():
        for qualifier in qualifiers:
            item = dict(qualifier)
            item['category'] = category
            qualifier_id = item.get('id')
            if isinstance(qualifier_id, int):
                by_id[qualifier_id] = item
            normalized = _normalize_name(item.get('name'))
            if normalized:
                by_name[normalized] = item

    return {
        'metadata': {k: v for k, v in data.items() if k not in {'categories'}},
        'by_id': by_id,
        'by_name': by_name,
    }


def qualifier_catalog_entry(qualifier_id: Any = None, name: Any = None) -> dict[str, Any] | None:
    catalog = load_opta_qualifier_catalog()

    parsed_id = None
    if qualifier_id is not None:
        try:
            parsed_id = int(qualifier_id)
        except (TypeError, ValueError):
            parsed_id = None

    if parsed_id is not None and parsed_id in catalog['by_id']:
        return catalog['by_id'][parsed_id]

    normalized = _normalize_name(name)
    if normalized:
        return catalog['by_name'].get(normalized)

    return None
