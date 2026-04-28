from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import soccerdata as sd
from pymongo import MongoClient, UpdateOne

from helper import setup_logging, patch_soccerdata, load_app_config, logger
from config import ScrapeDataConfig

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'


class GameSchedule:
    def __init__(self, config: ScrapeDataConfig):
        self.config = config
        patch_soccerdata()
        self.ws = sd.WhoScored(
            leagues=config.season.league,
            seasons=config.season.year,
        )
        self.client = MongoClient(config.mongo.url)
        self.db = self.client[config.mongo.db]
        self.collection_schedule = self.db[config.mongo.collection['collection_schedule']]
        self.collection_teams = self.db[config.mongo.collection['collection_teams']]

    def _read_season_schedule(self) -> pd.DataFrame:
        df = self.ws.read_schedule(force_cache=False)
        return df.reset_index(drop=False)

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        required = set(self.config.schedule_games.required_columns)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"WhoScored schedule is missing required columns: {sorted(missing)}")

    def _prepare_finished_schedule(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_required_columns(df)
        finished = df[df['status'] == self.config.schedule_games.finished_status_code].copy()
        if finished.empty:
            return finished

        if 'start_time' in finished.columns:
            finished['game_date'] = pd.to_datetime(finished['start_time'], errors='coerce')
        else:
            finished['game_date'] = pd.NaT

        return finished

    def _build_docs(self, df: pd.DataFrame) -> list[dict]:
        docs: list[dict] = []
        for row in df.to_dict(orient='records'):
            start_time = row.get('game_date')
            if pd.notna(start_time):
                if hasattr(start_time, 'to_pydatetime'):
                    start_time = start_time.to_pydatetime()
            else:
                start_time = None

            docs.append({
                'game_id': int(row['game_id']),
                'game_date': start_time,
                'season': self.config.season.year,
                'week': int(row.get('week')) if row.get('week') is not None and str(row.get('week')).isdigit() else None,
                'competition_name': self.config.season.name,
                'competition_country': self.config.season.country,
                'home_team_id': row.get('home_team_id'),
                'home_team_name': row.get('home_team'),
                'away_team_id': row.get('away_team_id'),
                'away_team_name': row.get('away_team'),
                'game_status': 'finished',
            })
        return docs

    def save_schedule(self) -> Dict[str, int]:
        schedule_df = self._read_season_schedule()
        finished_df = self._prepare_finished_schedule(schedule_df)
        if finished_df.empty:
            logger.info('No finished games found in schedule.')
            self.client.close()
            return {'processed': 0, 'upserted': 0}

        docs = self._build_docs(finished_df)
        ops = [
            UpdateOne({'game_id': doc['game_id']}, {'$set': doc}, upsert=True)
            for doc in docs
        ]

        result = self.collection_schedule.bulk_write(ops, ordered=False)
        affected = int(result.upserted_count + result.modified_count)
        logger.info('Schedule saved: %s finished games, %s affected docs', len(docs), affected)
        self.client.close()
        return {'processed': len(docs), 'upserted': affected}


def main(config_path: Path | str = CONFIG_PATH) -> Dict[str, int]:
    setup_logging()
    config = load_app_config(str(config_path))
    schedule = GameSchedule(config)
    return schedule.save_schedule()


if __name__ == '__main__':
    main()
