"""Publish season data from the local database to the Bundesliga app's Atlas database."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import Iterable

from pymongo import MongoClient, UpdateOne
from pymongo.errors import PyMongoError
from pymongo.write_concern import WriteConcern

from prod_pipeline.app_projections import (
    SOURCE_EVENTS,
    SOURCE_PASSES,
    SOURCE_PLAYER_STATS,
    SOURCE_SHOTS,
    SOURCE_TEAM_STATS,
    TARGET_EVENTS,
    TARGET_PASSES,
    TARGET_PLAYER_STATS,
    TARGET_SHOTS,
    TARGET_TEAM_STATS,
    app_event,
    app_player_stats,
    app_team_stats,
    sequence_documents,
)
from prod_pipeline.config import ScrapeDataConfig
from prod_pipeline.indexes import ensure_application_indexes

ATLAS_DATABASE = "WhoScored"
BATCH_SIZE = 500


@dataclass(frozen=True)
class WriteCounts:
    added: int = 0
    updated: int = 0

    @property
    def total(self) -> int:
        return self.added + self.updated

    def __add__(self, other: WriteCounts) -> WriteCounts:
        return WriteCounts(self.added + other.added, self.updated + other.updated)


def upsert_documents(collection, documents: Iterable[dict], key_fields: tuple[str, ...], *,
                     insert_only: bool = False) -> WriteCounts:
    count = WriteCounts()
    operator = "$setOnInsert" if insert_only else "$set"
    remaining = iter(documents)
    while batch := list(islice(remaining, BATCH_SIZE)):
        operations = []
        for document in batch:
            identity = {field: document[field] for field in key_fields}
            fields = {key: value for key, value in document.items() if key != "_id"}
            operations.append(UpdateOne(identity, {operator: fields}, upsert=True))
        result = collection.bulk_write(operations, ordered=False)
        count += WriteCounts(result.upserted_count, result.modified_count)
    return count


def project_documents(collection, query: dict, project, identity: tuple[str, ...]):
    for row in collection.find(query, {"_id": 0}):
        document = project(row)
        document["season"] = row["season"]
        document["_id"] = ":".join(str(row[field]) for field in identity)
        yield document


def publish_app_projections(source, target, season: str, game_ids: Iterable[int]) -> dict[str, WriteCounts]:
    """Refresh the season’s app collections from derived match data."""
    query = {"season": season}
    counts: dict[str, WriteCounts] = {}
    game_ids = tuple(game_ids)
    for source_name, target_name, project, identity in (
        (SOURCE_EVENTS, TARGET_EVENTS, app_event, ("game_id", "event_idx")),
        (SOURCE_TEAM_STATS, TARGET_TEAM_STATS, app_team_stats, ("game_id", "team_id")),
        (SOURCE_PLAYER_STATS, TARGET_PLAYER_STATS, app_player_stats, ("game_id", "player_id")),
    ):
        counts[target_name] = upsert_documents(
            target[target_name],
            project_documents(source[source_name], query, project, identity),
            ("_id",),
        )

    for source_name, target_name in ((SOURCE_SHOTS, TARGET_SHOTS), (SOURCE_PASSES, TARGET_PASSES)):
        counts[target_name] = WriteCounts()
        for game_id in game_ids:
            rows = source[source_name].find({**query, "game_id": game_id}, {"_id": 0})
            counts[target_name] += upsert_documents(
                target[target_name], sequence_documents(rows), ("_id",),
            )
    return counts


def publish_season(config: ScrapeDataConfig, atlas_uri: str, *, schedule_only: bool = False) -> dict[str, WriteCounts]:
    if atlas_uri.rstrip("/") == config.mongo.url.rstrip("/"):
        raise ValueError("Atlas publication requires a destination separate from the local database.")

    query = {"season": config.season.year}
    collection_names = config.mongo.collections
    counts = {}
    try:
        with (
            MongoClient(config.mongo.url, serverSelectionTimeoutMS=10000) as local_client,
            MongoClient(atlas_uri, serverSelectionTimeoutMS=10000) as atlas_client,
        ):
            source = local_client[config.mongo.db]
            target = atlas_client.get_database(ATLAS_DATABASE, write_concern=WriteConcern("majority"))
            schedule = source[collection_names["collection_schedule"]]
            game_ids = schedule.distinct("game_id", query)
            if not game_ids:
                raise ValueError(f"No local fixtures found for season {config.season.year}.")

            counts["game_schedule"] = upsert_documents(
                target.game_schedule, schedule.find(query), ("game_id",),
            )
            counts["season_config"] = upsert_documents(
                target.season_config, source.season_config.find(query), ("season",), insert_only=True,
            )
            if schedule_only:
                ensure_application_indexes(target)
                return counts

            for source_key, target_name, identity in (
                ("collection_formations", "game_formations", ("season", "game_id", "team_id")),
                ("collection_lineups", "game_lineups", ("season", "game_id", "team_id", "player_id")),
                ("collection_key_events", "game_key_events", ("game_id", "event_id")),
            ):
                counts[target_name] = upsert_documents(
                    target[target_name], source[collection_names[source_key]].find(query), identity,
                )

            counts.update(publish_app_projections(source, target, config.season.year, game_ids))
            ensure_application_indexes(target)
    except PyMongoError as error:
        code = getattr(error, "code", None)
        detail = f"{type(error).__name__}, code {code}" if code is not None else type(error).__name__
        raise RuntimeError(f"Atlas publication failed ({detail}). Retry with --atlas-only.") from None
    return counts
