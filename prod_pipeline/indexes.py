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


# These lookups can reuse an index with additional trailing keys.
QUERY_INDEXES = {
    "game_schedule": [
        (("game_id", 1), ("competition_name", 1)),
        (("competition_name", 1), ("competition_country", 1), ("season", 1), ("game_status", 1), ("game_date", -1)),
        (("competition_name", 1), ("competition_country", 1), ("game_status", 1), ("game_date", -1)),
    ],
    "historical_results": [(("season", 1), ("game_date", -1))],
    "season_config": [(("season", 1),)],
    "game_key_events": [
        (("game_id", 1), ("event_type", 1), ("minute", 1)),
        (("game_id", 1), ("minute", 1), ("second", 1), ("event_id", 1)),
    ],
    "game_lineups": [(("game_id", 1), ("competition_name", 1))],
    "game_formations": [(("game_id", 1), ("competition_name", 1))],
    "formation_coordinates": [(("formation_key", 1),), (("formation_name", 1),)],
    "game_processed_events_app": [
        (("game_id", 1), ("shot_goal", 1)),
        (("game_id", 1), ("player_id", 1)),
    ],
    "game_pass_sequences_app": [(("game_id", 1), ("sequence_id", 1), ("sequence_key", 1))],
    "game_shot_sequences_app": [(("game_id", 1), ("sequence_id", 1), ("sequence_key", 1))],
    "game_raw_events": [
        (("season", 1), ("game_id", 1)),
        (("game_id", 1),),
    ],
}


def ensure_application_indexes(db: Database) -> dict[str, list[str]]:
    ensured: dict[str, list[str]] = {}
    existing = set(db.list_collection_names(filter={"type": "collection"}))
    for collection_name in (INDEXES | QUERY_INDEXES).keys() & existing:
        collection = db[collection_name]
        current_indexes = {
            tuple(index["key"].items()): index["name"]
            for index in collection.list_indexes()
            if not any(index.get(option) for option in (
                "hidden", "sparse", "partialFilterExpression", "collation",
            ))
        }
        exact = INDEXES.get(collection_name, [])
        lookups = QUERY_INDEXES.get(collection_name, [])
        names: list[str] = []
        for fields in exact + lookups:
            name = current_indexes.get(fields)
            if name is None and fields in lookups:
                name = next(
                    (name for key, name in current_indexes.items() if key[:len(fields)] == fields),
                    None,
                )
            if name is None:
                name = collection.create_index(fields)
                current_indexes[fields] = name
            names.append(name)
        ensured[collection_name] = names
    return ensured
