from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict
from zipfile import ZipFile, ZIP_DEFLATED

from bson import json_util
from pymongo import MongoClient

from prod_pipeline.app_projections import (
    TARGET_EVENTS, TARGET_PASSES, TARGET_PLAYER_STATS, TARGET_SHOTS, TARGET_TEAM_STATS,
)
from prod_pipeline.helper import setup_logging, load_app_config, logger
from prod_pipeline.config import ScrapeDataConfig

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'


APP_BACKUP_COLLECTIONS = {
    'processed_events_app': TARGET_EVENTS,
    'shot_sequences_app': TARGET_SHOTS,
    'pass_sequences_app': TARGET_PASSES,
    'team_game_stats_app': TARGET_TEAM_STATS,
    'player_game_stats_app': TARGET_PLAYER_STATS,
}


def missing_app_backups(config: ScrapeDataConfig) -> bool:
    root = Path(config.mongo.backup_folder)
    return any(
        not (root / f"{config.season.year}_{name}.zip").is_file()
        for name in APP_BACKUP_COLLECTIONS
    )


def backup_season_data(config: ScrapeDataConfig) -> Dict[str, int]:
    backup_root = Path(config.mongo.backup_folder)
    backup_root.mkdir(parents=True, exist_ok=True)

    season = config.season.year
    collections = {
        'schedule': config.mongo.collections['collection_schedule'],
        'lineups': config.mongo.collections.get('collection_lineups', 'game_lineups'),
        'formations': config.mongo.collections.get('collection_formations', 'game_formations'),
        'raw_events': config.mongo.collections['collection_raw_events'],
        'processed_events': config.mongo.collections['collection_processed_events'],
        'key_events': config.mongo.collections.get('collection_key_events', 'game_key_events'),
        'team_game_stats': config.mongo.collections.get('collection_team_game_stats', 'game_team_stats'),
        'player_game_stats': config.mongo.collections.get('collection_player_game_stats', 'game_player_stats'),
        'shot_sequences': config.mongo.collections.get('collection_shot_sequences', 'game_shot_sequences'),
        'pass_sequences': config.mongo.collections.get('collection_pass_sequences', 'game_pass_sequences'),
        **APP_BACKUP_COLLECTIONS,
    }

    exported_count = 0
    with MongoClient(config.mongo.url) as client:
        db = client[config.mongo.db]
        for name, coll_name in collections.items():
            out_zip = backup_root / f"{season}_{name}.zip"
            with db[coll_name].find({'season': season}) as cursor:
                first = next(cursor, None)
                if first is None:
                    logger.info('No records found for season %s in collection %s', season, coll_name)
                    continue

                with NamedTemporaryFile(dir=backup_root, prefix=out_zip.name + '.', delete=False) as temporary:
                    temp_path = Path(temporary.name)
                try:
                    count = 1
                    with ZipFile(temp_path, 'w', compression=ZIP_DEFLATED) as archive:
                        with archive.open(f"{season}_{name}.json", 'w', force_zip64=True) as entry:
                            entry.write(b'[\n')
                            entry.write(json_util.dumps(first).encode('utf-8'))
                            for document in cursor:
                                entry.write(b',\n')
                                entry.write(json_util.dumps(document).encode('utf-8'))
                                count += 1
                            entry.write(b'\n]\n')
                    temp_path.replace(out_zip)
                finally:
                    temp_path.unlink(missing_ok=True)
                logger.info('Wrote %s records for %s into %s', count, name, out_zip)
                exported_count += count
    return {'exported_count': exported_count}


def main(config_path: Path | str = CONFIG_PATH) -> Dict[str, int]:
    setup_logging()
    config = load_app_config(str(config_path))
    return backup_season_data(config)


if __name__ == '__main__':
    main()
