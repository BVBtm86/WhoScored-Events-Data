"""Build traceable, normalized key-match events from WhoScored processed events.

The source is never modified.  A game state document is promoted only after every
derived event has been written, so clients can query the fingerprint named by the
latest completed state and never see a partial regeneration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pymongo import ASCENDING, MongoClient

# Keep the documented direct command (`python prod_pipeline/process_game_key_events.py`)
# working as well as module execution from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prod_pipeline.helper import load_app_config, setup_logging

CLASSIFICATION_VERSION = "1.3"
SOURCE_COLLECTION = "game_processed_events"
DESTINATION = "game_key_events"
STATE_COLLECTION = "game_key_event_state"
LOG = logging.getLogger(__name__)


def fingerprint(events: list[dict[str, Any]]) -> str:
    """Stable fingerprint of the event fields used by classification."""
    encoded = json.dumps(events, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def classify_event(event: dict[str, Any]) -> tuple[str, str | None] | None:
    """Classify only the flags written by ProcessedEvents from source qualifiers."""
    # Own goals are represented by ``shot_own_goal`` while ``shot_goal`` is
    # deliberately false (the shot was not scored for the shooting team).
    # Check this flag first so the event is still promoted to the match
    # timeline and retains its football-specific subtype.
    if event.get("shot_own_goal"):
        return "goal", "own_goal"
    if event.get("shot_goal"):
        return "goal", "penalty" if event.get("shot_penalty") else "open_play_or_other"
    if event.get("shot_penalty") and not event.get("shot_goal"):
        return "missed_penalty", None
    if event.get("second_yellow_card"):
        return "card", "second_yellow"
    if event.get("red_card"):
        return "card", "red"
    if event.get("yellow_card"):
        return "card", "yellow"
    if event.get("substitution_on"):
        return "substitution", "on"
    return None


def normalized_minute(event: dict[str, Any]) -> tuple[int | None, int | None]:
    """WhoScored's expanded_minute is the display minute; added time is retained only when explicit."""
    minute = event.get("expanded_minute")
    if not isinstance(minute, int):
        minute = event.get("minute") if isinstance(event.get("minute"), int) else None
    # Source documents do not carry an independently reliable stoppage-time field.
    return minute, None


def normalize_game_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    players = {event.get("player_id"): event.get("player") for event in events if event.get("player_id") is not None}
    normalized: list[dict[str, Any]] = []
    ambiguous = 0
    for event in events:
        try:
            classification = classify_event(event)
            if not classification:
                continue
            event_type, event_subtype = classification
            game_id = event.get("game_id")
            if not isinstance(game_id, int) or not isinstance(event.get("team_id"), int):
                ambiguous += 1
                continue
            minute, added_time = normalized_minute(event)
            related_player_id = event.get("related_player_id") if event_type == "substitution" else None
            assist_player_id = event.get("related_player_id") if event_type == "goal" and event.get("goal_assist") else None
            normalized.append({
                "game_id": game_id,
                "game_date": event.get("game_date"),
                "season": event.get("season"), "week": event.get("week"), "team_id": event.get("team_id"),
                "player_id": event.get("player_id"), "event_id": event.get("event_idx"),
                "event_type": event_type, "event_subtype": event_subtype,
                "minute": minute, "second": event.get("second"),
                "added_time": added_time, "period": event.get("period"),
                "player_name": event.get("player"), "team_name": event.get("team"),
                "related_player_id": related_player_id,
                "related_player_name": players.get(related_player_id),
                "assist_player_id": assist_player_id,
                "assist_player_name": players.get(assist_player_id),
                "source_event_id": event.get("event_idx"),
            })
        except (TypeError, ValueError):
            ambiguous += 1
    return normalized, ambiguous


class KeyEventProcessor:
    def __init__(self, uri: str, database: str = "WhoScored", batch_size: int = 100):
        self.client = MongoClient(uri)
        self.db = self.client[database]
        self.source = self.db[SOURCE_COLLECTION]
        self.destination = self.db[DESTINATION]
        self.state = self.db[STATE_COLLECTION]
        self.batch_size = batch_size

    def ensure_indexes(self) -> None:
        # Migrate the first, single-version index if an earlier development run
        # created it. Versioned documents need the fingerprint in the key.
        for old_index in ("unique_event_identity", "unique_event_identity_version"):
            try:
                self.destination.drop_index(old_index)
            except Exception:
                pass
        self.destination.create_index(
            [("game_id", ASCENDING), ("event_id", ASCENDING)],
            unique=True,
            name="unique_event_identity",
        )
        self.destination.create_index([("game_id", ASCENDING)], name="game_lookup")
        self.destination.create_index([("game_id", ASCENDING), ("minute", ASCENDING), ("event_id", ASCENDING)], name="match_chronology")
        self.state.create_index([("game_id", ASCENDING)], unique=True, name="unique_game_state")

    def game_ids(self, game_id: int | None = None, season: str | None = None) -> Iterable[int]:
        if game_id is not None:
            yield game_id
            return
        match = {"competition_name": "Bundesliga", "game_status": "finished"}
        if season:
            match["season"] = season
        cursor = self.source.aggregate([
            {"$match": match},
            {"$group": {"_id": "$game_id"}}, {"$sort": {"_id": 1}},
        ], allowDiskUse=True)
        for row in cursor:
            if isinstance(row.get("_id"), int):
                yield row["_id"]

    def source_events(self, game_id: int) -> list[dict[str, Any]]:
        projection = {"_id": 0, "game_id": 1, "game_date": 1, "season": 1, "week": 1, "team_id": 1, "team": 1,
                      "player_id": 1, "player": 1, "event_idx": 1, "minute": 1, "expanded_minute": 1,
                      "second": 1,
                      "period": 1, "related_player_id": 1, "goal_assist": 1, "shot_goal": 1, "shot_own_goal": 1,
                      "shot_penalty": 1, "yellow_card": 1, "second_yellow_card": 1, "red_card": 1,
                      "substitution_on": 1, "game_status": 1}
        return list(self.source.find({"game_id": game_id, "competition_name": "Bundesliga"}, projection).sort([("event_idx", 1)]))

    def process_game(self, game_id: int, force: bool = False) -> dict[str, int | str]:
        events = self.source_events(game_id)
        if not events:
            return {"status": "skipped", "events": 0, "ambiguous": 0}
        source_hash = fingerprint(events)
        state = self.state.find_one({"game_id": game_id}, {"source_fingerprint": 1})
        if (
            not force
            and state
            and state.get("source_fingerprint") == source_hash
            and state.get("classification_version") == CLASSIFICATION_VERSION
        ):
            return {"status": "skipped", "events": 0, "ambiguous": 0}
        docs, ambiguous = normalize_game_events(events)
        for document in docs:
            self.destination.replace_one(
                {"game_id": document["game_id"], "event_id": document["event_id"]},
                document,
                upsert=True,
            )
        self.state.replace_one({"game_id": game_id}, {"game_id": game_id, "source_fingerprint": source_hash,
                                                       "event_count": len(docs), "classification_version": CLASSIFICATION_VERSION,
                                                       "completed_at": datetime.now(timezone.utc)}, upsert=True)
        valid_ids = [document["event_id"] for document in docs]
        self.destination.delete_many({"game_id": game_id, "event_id": {"$nin": valid_ids}})
        return {"status": "processed", "events": len(docs), "ambiguous": ambiguous}

    def run(self, mode: str, game_id: int | None = None, season: str | None = None,
            stop_on_error: bool = False) -> dict[str, int]:
        self.ensure_indexes()
        totals = {"processed": 0, "skipped": 0, "failed": 0, "events": 0, "ambiguous": 0}
        for index, candidate in enumerate(self.game_ids(game_id, season), start=1):
            try:
                result = self.process_game(candidate, force=mode == "backfill")
                totals[str(result["status"])] += 1
                totals["events"] += int(result["events"])
                totals["ambiguous"] += int(result["ambiguous"])
            except Exception:
                totals["failed"] += 1
                LOG.exception("Failed game_id=%s", candidate)
                if stop_on_error:
                    raise
            if index % self.batch_size == 0:
                LOG.info("Progress: %s games; processed=%s skipped=%s failed=%s", index, totals["processed"], totals["skipped"], totals["failed"])
        return totals


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize key WhoScored events for match previews.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--backfill", action="store_true")
    group.add_argument("--incremental", action="store_true")
    group.add_argument("--game-id", type=int)
    parser.add_argument("--mongo-uri")
    parser.add_argument("--database")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()
    setup_logging()
    config = load_app_config("config/config.yaml")
    processor = KeyEventProcessor(args.mongo_uri or config.mongo.url, args.database or config.mongo.db, args.batch_size)
    mode = "backfill" if args.backfill else "incremental" if args.incremental else "single"
    totals = processor.run(mode, args.game_id)
    LOG.info("Key event processing complete: %s", totals)


if __name__ == "__main__":
    main()
