"""Read-only validation report for sequence origin coverage.

Usage: run with the normal project interpreter after sequence processing has
completed. The script never updates sequence or source-event collections.
"""
from collections import Counter
import argparse
from pymongo import MongoClient


def report(uri: str, database: str, season: str | None = None) -> dict:
    client = MongoClient(uri)
    db = client[database]
    output = {}
    try:
        for collection_name in ("game_shot_sequences", "game_pass_sequences"):
            query = {"season": season} if season else {}
            rows = db[collection_name].find(
                query,
                {
                    "_id": 0,
                    "sequence_key": 1,
                    "sequence_event": 1,
                    "origin_type": 1,
                    "origin_event_id": 1,
                    "origin_evidence": 1,
                },
            )
            sequences = {}
            for row in rows:
                key = row.get("sequence_key")
                if key is None:
                    continue
                current = sequences.setdefault(key, row)
                if current.get("origin_type") is None and row.get("origin_type") is not None:
                    sequences[key] = row
            categories = Counter(
                row.get("origin_type") or "Unresolved"
                for row in sequences.values()
            )
            output[collection_name] = {
                "total_sequences": len(sequences),
                "classified_sequences": len(sequences) - categories.get("Unresolved", 0),
                "unresolved_sequences": categories.get("Unresolved", 0),
                "coverage_percent": round(
                    100 * (len(sequences) - categories.get("Unresolved", 0)) / len(sequences), 2
                )
                if sequences
                else 0.0,
                "by_origin": dict(sorted(categories.items())),
            }
        return output
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="mongodb://127.0.0.1:27018/")
    parser.add_argument("--database", default="WhoScored")
    parser.add_argument("--season")
    args = parser.parse_args()
    import json

    print(json.dumps(report(args.uri, args.database, args.season), indent=2, ensure_ascii=False))
