from __future__ import annotations

from pathlib import Path
from typing import Dict
from zipfile import ZipFile, ZIP_DEFLATED

from bson import json_util
from pymongo import MongoClient

from helper import setup_logging, load_app_config, logger
from config import ScrapeDataConfig

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'


def backup_season_data(config: ScrapeDataConfig) -> Dict[str, int]:
    backup_root = Path(config.mongo.backup_folder)
    backup_root.mkdir(parents=True, exist_ok=True)

    client = MongoClient(config.mongo.url)
    db = client[config.mongo.db]

    season = config.season.year
    collections = {
        'schedule': config.mongo.collections['collection_schedule'],
        'raw_events': config.mongo.collections['collection_raw_events'],
        'processed_events': config.mongo.collections['collection_processed_events'],
        'team_game_stats': config.mongo.collections.get('collection_team_game_stats', 'game_team_stats'),
        'player_game_stats': config.mongo.collections.get('collection_player_game_stats', 'game_player_stats'),
        'shot_sequences': config.mongo.collections.get('collection_shot_sequences', 'game_shot_sequences'),
        'pass_sequences': config.mongo.collections.get('collection_pass_sequences', 'game_pass_sequences'),
    }

    exported_count = 0
    for name, coll_name in collections.items():
        docs = list(db[coll_name].find({'season': season}))
        if not docs:
            logger.info('No records found for season %s in collection %s', season, coll_name)
            continue

        out_zip = backup_root / f"{season}_{name}.zip"
        payload = json_util.dumps(docs, indent=2)

        with ZipFile(out_zip, 'w', compression=ZIP_DEFLATED) as zf:
            zf.writestr(f"{season}_{name}.json", payload.encode('utf-8'))

        logger.info('Wrote %s records for %s into %s', len(docs), name, out_zip)
        exported_count += len(docs)

    client.close()
    return {'exported_count': exported_count}


def main(config_path: Path | str = CONFIG_PATH) -> Dict[str, int]:
    setup_logging()
    config = load_app_config(str(config_path))
    return backup_season_data(config)


if __name__ == '__main__':
    main()
