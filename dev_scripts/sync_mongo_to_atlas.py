from __future__ import annotations

import os

from pymongo import IndexModel, MongoClient, ReplaceOne


LOCAL_MONGO_URL = "mongodb://localhost:27017/"
DB_NAME = "WhoScored"
SEASON = "2025-2026"
BATCH_SIZE = 500
SYNC_INDEXES = True


def build_atlas_url() -> str:
    username = os.getenv("MONGO_USERNAME")
    password = os.getenv("MONGO_PASSWORD")
    host = os.getenv("MONGO_HOST")

    missing = [
        name
        for name, value in {
            "MONGO_USERNAME": username,
            "MONGO_PASSWORD": password,
            "MONGO_HOST": host,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    return (
        f"mongodb+srv://{username}:{password}@{host}/"
        "?retryWrites=true&w=majority&appName=Cluster0"
    )


COLLECTIONS = [
    "available_teams",
    "game_schedule",
    "game_lineups",
    "game_formations",
    "game_team_stats",
    "game_player_stats",
    # "game_raw_events",
    # "game_processed_events",
    # "game_shot_sequences",
    # "game_pass_sequences",
]


def season_filter(collection_name: str) -> dict:
    if collection_name == "available_teams":
        return {}
    return {"season": SEASON}


def sync_indexes(local_collection, atlas_collection) -> None:
    models: list[IndexModel] = []

    for index in local_collection.list_indexes():
        if index["name"] == "_id_":
            continue

        options = {
            key: index[key]
            for key in (
                "name",
                "unique",
                "sparse",
                "partialFilterExpression",
                "expireAfterSeconds",
                "collation",
            )
            if key in index
        }
        models.append(IndexModel(list(index["key"].items()), **options))

    if not models:
        print("  indexes: no custom indexes to sync")
        return

    created = atlas_collection.create_indexes(models)
    print(f"  indexes: ensured {len(created)} indexes")


def copy_collection(local_db, atlas_db, collection_name: str) -> None:
    print(f"\nCopying {collection_name}...")

    local_collection = local_db[collection_name]
    atlas_collection = atlas_db[collection_name]
    query = season_filter(collection_name)
    total = local_collection.count_documents(query)

    if total == 0:
        print("  0 docs, skipped")
        if SYNC_INDEXES:
            sync_indexes(local_collection, atlas_collection)
        return

    if SYNC_INDEXES:
        sync_indexes(local_collection, atlas_collection)

    batch = []
    copied = 0
    cursor = local_collection.find(query).batch_size(BATCH_SIZE)

    for doc in cursor:
        batch.append(ReplaceOne({"_id": doc["_id"]}, doc, upsert=True))

        if len(batch) >= BATCH_SIZE:
            atlas_collection.bulk_write(batch, ordered=False)
            copied += len(batch)
            print(f"  copied {copied}/{total}")
            batch = []

    if batch:
        atlas_collection.bulk_write(batch, ordered=False)
        copied += len(batch)
        print(f"  copied {copied}/{total}")


def main() -> None:
    atlas_mongo_url = build_atlas_url()
    local_client = MongoClient(LOCAL_MONGO_URL)
    atlas_client = MongoClient(atlas_mongo_url)

    try:
        local_db = local_client[DB_NAME]
        atlas_db = atlas_client[DB_NAME]

        for collection_name in COLLECTIONS:
            copy_collection(local_db, atlas_db, collection_name)

        print("\nAtlas databases:", atlas_client.list_database_names())
        print("Atlas WhoScored collections:", atlas_db.list_collection_names())
        print("\nDone.")
    finally:
        local_client.close()
        atlas_client.close()


if __name__ == "__main__":
    main()
