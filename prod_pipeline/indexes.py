"""Indexes required by the source and app-facing MongoDB collections."""

from __future__ import annotations

from pymongo.database import Database


INDEXES = {
    "game_processed_events": [
        (("game_id", 1), ("event_idx", 1)),
    ],
    "game_processed_events_app": [
        (("game_id", 1), ("event_idx", 1)),
        (("game_id", 1), ("categories", 1), ("event_idx", 1)),
    ],
    "game_shot_sequences": [
        (("game_id", 1), ("sequence_id", 1), ("event_idx", 1)),
    ],
    "game_shot_sequences_app": [
        (("game_id", 1),),
        (("game_id", 1), ("sequence_id", 1)),
    ],
    "game_pass_sequences": [
        (("game_id", 1), ("sequence_id", 1), ("event_idx", 1)),
    ],
    "game_pass_sequences_app": [
        (("game_id", 1),),
        (("game_id", 1), ("sequence_id", 1)),
    ],
    "game_team_stats": [
        (("game_id", 1), ("team_id", 1)),
    ],
    "game_team_stats_app": [
        (("game_id", 1), ("team_id", 1)),
    ],
    "game_player_stats": [
        (("game_id", 1), ("player_id", 1)),
    ],
    "game_player_stats_app": [
        (("game_id", 1), ("player_id", 1)),
    ],
}


def ensure_application_indexes(db: Database) -> dict[str, list[str]]:
    created: dict[str, list[str]] = {}
    existing = set(db.list_collection_names())
    for collection_name, specifications in INDEXES.items():
        if collection_name not in existing:
            continue
        names: list[str] = []
        for fields in specifications:
            name = db[collection_name].create_index(fields)
            names.append(name)
        created[collection_name] = names
    return created
