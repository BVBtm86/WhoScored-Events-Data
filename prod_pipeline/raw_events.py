from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd
import soccerdata as sd
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

from helper import setup_logging, patch_soccerdata, load_app_config, logger
from config import ScrapeDataConfig

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'


class RawEvents:
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
        self.collection_raw_events = self.db[config.mongo.collection['collection_raw_events']]

    def _finished_games_df(self) -> pd.DataFrame:
        docs = list(
            self.collection_schedule.find(
                {'season': self.config.season.year, 'game_status': 'finished'},
                {'_id': 0},
            )
        )
        return pd.DataFrame(docs)

    def _raw_events_extracted(self) -> Set[int]:
        return {int(x) for x in self.collection_raw_events.distinct('game_id') if x is not None}

    @staticmethod
    def _normalize_mongo_value(value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        return value

    def _process_raw_events(self, game_info: pd.DataFrame, events: pd.DataFrame) -> list[dict]:
        events = events.reset_index(drop=False)
        if 'event_idx' not in events.columns:
            if 'index' in events.columns:
                events = events.rename(columns={'index': 'event_idx'})
            else:
                events['event_idx'] = range(len(events))

        for field in ['league', 'season', 'game']:
            if field in events.columns:
                events = events.drop(columns=[field])

        if 'game_id' in events.columns:
            events['game_id'] = events['game_id'].astype(int)

        if 'game_id' not in game_info.columns:
            raise ValueError('Schedule metadata missing game_id')

        game_info = game_info.copy()
        game_info['game_id'] = game_info['game_id'].astype(int)

        merged = pd.merge(game_info, events, on='game_id', how='right')
        return [
            {k: self._normalize_mongo_value(v) for k, v in record.items()}
            for record in merged.to_dict(orient='records')
        ]

    def save_new_finished_games(self, limit: Optional[int] = None) -> Dict[str, int]:
        finished_df = self._finished_games_df()
        if finished_df.empty:
            logger.info('No finished schedule games found for raw event extraction.')
            self.client.close()
            return {'processed_games': 0, 'inserted_rows': 0}

        extracted_ids = self._raw_events_extracted()
        game_ids = [int(x) for x in finished_df['game_id'].tolist()]
        pending = [gid for gid in game_ids if gid not in extracted_ids]
        if limit is not None:
            pending = pending[:limit]

        if not pending:
            logger.info('No new finished games to extract raw events for.')
            self.client.close()
            return {'processed_games': 0, 'inserted_rows': 0}

        inserted_games = 0
        inserted_rows_total = 0

        for game_id in pending:
            try:
                logger.info('Fetching raw events for game_id=%s', game_id)
                events = self.ws.read_events(match_id=game_id, output_fmt='events')
                if events is None or len(events) == 0:
                    logger.info('No events available for game_id=%s', game_id)
                    continue

                game_info = finished_df[finished_df['game_id'] == game_id].reset_index(drop=True)
                if game_info.empty:
                    logger.warning('Schedule metadata missing for game_id=%s', game_id)
                    continue

                records = self._process_raw_events(game_info=game_info, events=events)
                if not records:
                    logger.info('No merged raw event records for game_id=%s', game_id)
                    continue

                try:
                    result = self.collection_raw_events.insert_many(records, ordered=False)
                    inserted_count = len(result.inserted_ids)
                except BulkWriteError as bwe:
                    inserted_count = bwe.details.get('nInserted', 0)
                    logger.warning(
                        'BulkWriteError for game_id=%s, inserted %s rows before failure',
                        game_id,
                        inserted_count,
                    )

                inserted_games += 1
                inserted_rows_total += inserted_count
                logger.info('Inserted %s rows for game_id=%s', inserted_count, game_id)

            except Exception:
                logger.exception('Error processing raw events for game_id=%s', game_id)

        self.client.close()
        return {'processed_games': len(pending), 'inserted_rows': inserted_rows_total}


def main(config_path: Path | str = CONFIG_PATH) -> Dict[str, int]:
    setup_logging()
    config = load_app_config(str(config_path))
    events = RawEvents(config)
    return events.save_new_finished_games()


if __name__ == '__main__':
    main()
