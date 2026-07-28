from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

from prod_pipeline.config import ScrapeDataConfig
from prod_pipeline.helper import load_app_config, logger, setup_logging

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'


class SequenceData:

    def __init__(self, config: ScrapeDataConfig):
        self.config = config
        self.client = MongoClient(config.mongo.url)
        self.db = self.client[config.mongo.db]
        self.collection_processed_events = self.db[config.mongo.collections['collection_processed_events']]
        self.collection_shot_sequences = self.db[config.mongo.collections['collection_shot_sequences']]
        self.collection_pass_sequences = self.db[config.mongo.collections['collection_pass_sequences']]

        sequence_config = config.sequence_data
        self.base_features = sequence_config.base_features
        self.sequence_action_types = set(sequence_config.sequence_action_types)
        self.shot_types = set(sequence_config.shot_types)
        self.ignore_types = set(sequence_config.ignore_types)
        self.break_types = set(sequence_config.break_types)
        self.opponent_pressure_types = set(sequence_config.opponent_pressure_types)
        self.shot_feature_keywords = sequence_config.shot_feature_keywords
        self.pass_feature_keywords = sequence_config.pass_feature_keywords
        self.shot_metadata_features = sequence_config.shot_metadata_features
        self.pass_metadata_features = sequence_config.pass_metadata_features
        self.shared_excluded_features = set(sequence_config.shared_excluded_features)
        self.shot_excluded_features = set(sequence_config.shot_excluded_features)
        self.pass_excluded_features = set(sequence_config.pass_excluded_features)

    def _processed_game_ids(self) -> Set[int]:
        return {
            int(x)
            for x in self.collection_processed_events.distinct('game_id', {'season': self.config.season.year})
            if x is not None
        }

    def _sequence_game_ids(self) -> Set[int]:
        shot_ids = {
            int(x)
            for x in self.collection_shot_sequences.distinct('game_id', {'season': self.config.season.year})
            if x is not None
        }
        pass_ids = {
            int(x)
            for x in self.collection_pass_sequences.distinct('game_id', {'season': self.config.season.year})
            if x is not None
        }
        return shot_ids & pass_ids

    @staticmethod
    def _normalize_mongo_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return {k: SequenceData._normalize_mongo_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [SequenceData._normalize_mongo_value(v) for v in value]
        if isinstance(value, tuple):
            return [SequenceData._normalize_mongo_value(v) for v in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        return value

    @classmethod
    def _records_for_mongo(cls, df: pd.DataFrame) -> list[dict]:
        return [
            {k: cls._normalize_mongo_value(v) for k, v in record.items()}
            for record in df.to_dict(orient='records')
        ]

    def _load_processed_game_events(self, game_id: int) -> pd.DataFrame:
        docs = list(
            self.collection_processed_events.find(
                {'season': self.config.season.year, 'game_id': int(game_id)},
                {'_id': 0},
            )
        )
        return pd.DataFrame(docs)

    @staticmethod
    def _matches_keyword(column: str, keyword: str) -> bool:
        return (
            column == keyword
            or column.startswith(f'{keyword}_')
            or column.endswith(f'_{keyword}')
            or f'_{keyword}_' in column
        )

    def _feature_columns(
        self,
        sequences_df: pd.DataFrame,
        feature_keywords: list[str],
        metadata_features: list[str],
        excluded_features: set[str],
    ) -> list[str]:
        final_features = list(self.base_features)
        for keyword in feature_keywords:
            final_features.extend(
                [feature for feature in sequences_df.columns if self._matches_keyword(feature, keyword)]
            )
        final_features.extend(metadata_features)
        final_features = list(dict.fromkeys(final_features))
        blocked = self.shared_excluded_features | excluded_features
        return [col for col in final_features if col in sequences_df.columns and col not in blocked]

    @staticmethod
    def _system_sequence_ending(current_period: Optional[str]) -> str:
        return 'End of Period' if current_period == 'FirstHalf' else 'End of Game'

    def _sequence_start_reason(
        self,
        start_event: Optional[pd.Series],
        sequence_team_id: int,
        current_period: Optional[str],
    ) -> str:
        if start_event is None:
            return 'Start of Period' if current_period == 'FirstHalf' else 'Start of Game'

        event_type = start_event.get('type')
        outcome = start_event.get('outcome_type')
        event_team_id = start_event.get('team_id')

        if start_event.get('period') != current_period:
            return 'Start of Period'

        if event_type in {'SubstitutionOn', 'SubstitutionOff', 'Card'}:
            return 'After Stoppage'

        if event_type in {'Save', 'Claim', 'Punch', 'Smother', 'KeeperPickup', 'KeeperSweeper'}:
            return 'Goalkeeper Restart'

        if event_type in self.shot_types:
            return 'After Shot'

        if event_type in {'Foul', 'OffsideGiven', 'OffsideProvoked', 'OffsidePass'}:
            return 'After Stoppage'

        if event_team_id != sequence_team_id:
            if event_type in {'Dispossessed', 'Turnover', 'Error'}:
                return 'Opponent Error'
            if outcome == 'Unsuccessful':
                return 'Won Possession'
            return 'After Opponent Action'

        if event_type in {'Pass', 'BallTouch'} and outcome != 'Successful':
            return 'After Unsuccessful Pass'

        if event_type in {'Dispossessed', 'Turnover', 'Error'}:
            return 'After Turnover'

        return f'After {event_type}' if event_type else 'Sequence Start'

    @staticmethod
    def _event_metadata(event: Optional[pd.Series], prefix: str) -> Dict[str, Any]:
        fields = {
            f'{prefix}_event_idx': None,
            f'{prefix}_type': None,
            f'{prefix}_outcome_type': None,
            f'{prefix}_team_id': None,
            f'{prefix}_team': None,
            f'{prefix}_player_id': None,
            f'{prefix}_player': None,
        }
        if event is None:
            return fields

        return {
            f'{prefix}_event_idx': event.get('event_idx'),
            f'{prefix}_type': event.get('type'),
            f'{prefix}_outcome_type': event.get('outcome_type'),
            f'{prefix}_team_id': event.get('team_id'),
            f'{prefix}_team': event.get('team'),
            f'{prefix}_player_id': event.get('player_id'),
            f'{prefix}_player': event.get('player'),
        }

    def _shooting_valid_sequence_event(
        self,
        event: pd.Series,
        shooting_team_id: int,
        current_period: str,
    ) -> str:
        event_type = event['type']
        outcome = event.get('outcome_type')
        event_period = event.get('period')

        if event_period != current_period:
            return 'break'

        if event_type in self.ignore_types:
            return 'ignore'

        if event_type in self.break_types:
            return 'break'

        same_team = event['team_id'] == shooting_team_id

        if not same_team:
            if event_type == 'Error':
                return 'keep'
            if outcome == 'Unsuccessful':
                return 'keep'
            return 'break'

        if event_type in {'Dispossessed', 'Turnover', 'Error'}:
            return 'break'

        if event_type in {'Pass', 'TakeOn', 'BallTouch'}:
            return 'keep' if outcome == 'Successful' else 'break'

        if event_type in self.sequence_action_types:
            return 'break' if outcome == 'Unsuccessful' else 'keep'

        return 'ignore'

    def _get_shot_sequence(
        self,
        game_data: pd.DataFrame,
        shot_index: int,
        sequence_id: int,
    ) -> pd.DataFrame:
        shooting_team_id = game_data.loc[shot_index, 'team_id']
        current_period = game_data.loc[shot_index, 'period']
        sequence_rows = [game_data.loc[shot_index]]
        start_event = None
        j = shot_index - 1

        while j >= 0:
            event = game_data.loc[j]
            decision = self._shooting_valid_sequence_event(event, shooting_team_id, current_period)

            if decision == 'ignore':
                j -= 1
                continue

            if decision == 'keep':
                sequence_rows.append(event)
                j -= 1
                continue

            if decision == 'break':
                start_event = event
                break

        sequence_df = pd.DataFrame(sequence_rows).sort_index()
        sequence_df['sequence_id'] = sequence_id
        sequence_df['sequence_key'] = str(game_data.loc[shot_index, 'game_id']) + '_' + str(sequence_id)
        sequence_df['sequence_event'] = range(1, len(sequence_df) + 1)
        sequence_df['shot_event_idx'] = game_data.loc[shot_index, 'event_idx']
        sequence_df['sequence_start_reason'] = self._sequence_start_reason(
            start_event,
            shooting_team_id,
            current_period,
        )
        sequence_df = sequence_df.assign(**self._event_metadata(start_event, 'sequence_start'))
        sequence_df = sequence_df.assign(**self._event_metadata(game_data.loc[shot_index], 'sequence_end'))
        sequence_df['sequence_end_side'] = 'same_team'

        return sequence_df

    def build_shot_sequences_for_game(self, game_data: pd.DataFrame) -> pd.DataFrame:
        if game_data.empty:
            return pd.DataFrame()

        game_data = game_data.sort_values('event_idx').reset_index(drop=True)
        sequences = []
        sequence_id = 1

        shot_mask = game_data['shot_event'].fillna(False).astype(bool)
        shot_indexes = game_data[shot_mask].index

        for shot_index in shot_indexes:
            sequence_df = self._get_shot_sequence(game_data, shot_index, sequence_id)
            if not sequence_df.empty:
                sequences.append(sequence_df)
                sequence_id += 1

        if not sequences:
            return pd.DataFrame()

        sequences_df = pd.concat(sequences, ignore_index=True)
        return sequences_df[
            self._feature_columns(
                sequences_df,
                self.shot_feature_keywords,
                self.shot_metadata_features,
                self.shot_excluded_features,
            )
        ].reset_index(drop=True)

    def _passing_valid_sequence_event(
        self,
        event: pd.Series,
        passing_team_id: int,
        current_period: str,
    ) -> str:
        event_type = event['type']
        outcome = event.get('outcome_type')
        event_period = event.get('period')
        same_team = event['team_id'] == passing_team_id

        if event_period != current_period:
            return 'break'

        if event_type in self.ignore_types:
            return 'ignore'

        if event_type in {'OffsideGiven', 'OffsideProvoked'}:
            return 'ignore'

        if event_type == 'OffsidePass':
            return 'keep_break' if same_team else 'break'

        if not same_team:
            if event_type == 'Error':
                return 'keep'
            if outcome == 'Unsuccessful':
                return 'keep'
            if event_type == 'Pass':
                return 'break'
            return 'keep_break'

        if event_type == 'Error':
            return 'keep'

        if event_type in {'Dispossessed', 'Turnover'}:
            return 'keep_break'

        if event_type in self.shot_types:
            return 'break'

        if event_type in self.break_types:
            return 'keep_break'

        if event_type in {'Pass', 'BallTouch'} and outcome == 'Successful':
            return 'keep'

        if event_type in {'Pass', 'BallTouch'} and outcome != 'Successful':
            return 'keep_break'

        return 'ignore'

    def _passing_sequence_ending(
        self,
        event: pd.Series,
        current_team_id: int,
        current_period: str,
    ) -> str:
        event_type = event['type']
        outcome = event.get('outcome_type')
        same_team = event['team_id'] == current_team_id

        if event.get('period') != current_period:
            return self._system_sequence_ending(current_period)

        if event_type in self.ignore_types:
            return self._system_sequence_ending(current_period)

        if same_team and event_type == 'Goal':
            return 'Goal'

        if same_team and event_type in self.shot_types:
            return 'Shot'

        if same_team and event_type == 'Pass' and outcome != 'Successful':
            return 'Unsuccessful Pass'

        if same_team and event_type == 'BallTouch' and outcome != 'Successful':
            return 'Unsuccessful Ball Touch'

        if same_team and event_type in {'OffsideGiven', 'OffsidePass', 'OffsideProvoked'}:
            return 'Offside'

        if not same_team:
            if event_type in {'Pass', 'OffsidePass'}:
                return 'Loss Possession'
            return event_type

        return event_type

    def _add_passing_sequence_end_metadata(
        self,
        current_rows: list[pd.Series],
        start_event: Optional[pd.Series],
        current_team_id: int,
        event: pd.Series,
        sequence_ending: str,
        period_changed: bool = False,
    ) -> list[pd.Series]:
        sequence_team = current_rows[0].get('team')
        if sequence_ending == 'Loss Possession':
            same_team_rows = [row for row in current_rows if row.get('team_id') == current_team_id]
            sequence_end_event = same_team_rows[-1] if same_team_rows else current_rows[-1]
        else:
            sequence_end_event = event

        for row in current_rows:
            row['sequence_start_reason'] = self._sequence_start_reason(
                start_event,
                current_team_id,
                row.get('period'),
            )
            for key, value in SequenceData._event_metadata(start_event, 'sequence_start').items():
                row[key] = value
            row['sequence_team_id'] = current_team_id
            row['sequence_team'] = sequence_team
            row['sequence_ending'] = sequence_ending

            if period_changed:
                for key, value in SequenceData._event_metadata(None, 'sequence_end').items():
                    row[key] = value
                row['sequence_end_side'] = 'system'
            else:
                for key, value in SequenceData._event_metadata(sequence_end_event, 'sequence_end').items():
                    row[key] = value
                row['sequence_end_side'] = (
                    'same_team'
                    if sequence_ending == 'Loss Possession' or event.get('team_id') == current_team_id
                    else 'opponent'
                )

        return current_rows

    def build_passing_sequences_for_game(self, game_data: pd.DataFrame) -> pd.DataFrame:
        if game_data.empty:
            return pd.DataFrame()

        game_data = game_data.sort_values(['period', 'minute', 'second', 'event_idx']).reset_index(drop=True)

        sequences = []
        current_rows: list[pd.Series] = []
        current_team_id = None
        current_period = None
        current_start_event = None
        sequence_id = 1

        for idx, event in game_data.iterrows():
            event_type = event['type']
            outcome = event.get('outcome_type')

            if current_team_id is None:
                if event_type == 'Pass' and outcome == 'Successful':
                    current_team_id = event['team_id']
                    current_period = event['period']
                    current_start_event = game_data.loc[idx - 1] if idx > 0 else None

                    row = event.copy()
                    row['sequence_id'] = sequence_id
                    current_rows = [row]
                continue

            pass_decision = self._passing_valid_sequence_event(event, current_team_id, current_period)

            if pass_decision == 'keep':
                row = event.copy()
                row['sequence_id'] = sequence_id
                current_rows.append(row)
                continue

            if pass_decision == 'keep_break':
                sequence_ending = self._passing_sequence_ending(event, current_team_id, current_period)
                current_rows = self._add_passing_sequence_end_metadata(
                    current_rows=current_rows,
                    start_event=current_start_event,
                    current_team_id=current_team_id,
                    event=event,
                    sequence_ending=sequence_ending,
                )

                sequences.extend(current_rows)
                sequence_id += 1
                current_rows = []
                current_team_id = None
                current_period = None
                current_start_event = None
                continue

            if pass_decision == 'ignore':
                continue

            if pass_decision == 'break':
                sequence_ending = self._passing_sequence_ending(event, current_team_id, current_period)
                period_changed = event.get('period') != current_period

                current_rows = self._add_passing_sequence_end_metadata(
                    current_rows=current_rows,
                    start_event=current_start_event,
                    current_team_id=current_team_id,
                    event=event,
                    sequence_ending=sequence_ending,
                    period_changed=period_changed,
                )

                sequences.extend(current_rows)
                sequence_id += 1
                current_rows = []
                current_team_id = None
                current_period = None
                current_start_event = None

                if event_type == 'Pass' and outcome == 'Successful':
                    current_team_id = event['team_id']
                    current_period = event['period']
                    current_start_event = game_data.loc[idx - 1] if idx > 0 else None

                    row = event.copy()
                    row['sequence_id'] = sequence_id
                    current_rows = [row]

        if current_rows:
            sequence_ending = self._system_sequence_ending(current_period)

            for row in current_rows:
                row['sequence_start_reason'] = self._sequence_start_reason(
                    current_start_event,
                    current_team_id,
                    row.get('period'),
                )
                for key, value in self._event_metadata(current_start_event, 'sequence_start').items():
                    row[key] = value
                row['sequence_team_id'] = current_team_id
                row['sequence_team'] = current_rows[0].get('team')
                row['sequence_ending'] = sequence_ending
                row['sequence_end_event_idx'] = None
                row['sequence_end_type'] = None
                row['sequence_end_outcome_type'] = None
                row['sequence_end_team_id'] = None
                row['sequence_end_team'] = None
                row['sequence_end_player_id'] = None
                row['sequence_end_player'] = None
                row['sequence_end_side'] = 'system'

            sequences.extend(current_rows)

        sequences_df = pd.DataFrame(sequences)
        if sequences_df.empty:
            return sequences_df

        sequences_df['sequence_event'] = sequences_df.groupby('sequence_id').cumcount() + 1
        sequences_df['sequence_key'] = (
            sequences_df['game_id'].astype(str) + '_' + sequences_df['sequence_id'].astype(str)
        )

        return sequences_df[
            self._feature_columns(
                sequences_df,
                self.pass_feature_keywords,
                self.pass_metadata_features,
                self.pass_excluded_features,
            )
        ].reset_index(drop=True)

    def save_sequence_data(self, limit: Optional[int] = None) -> Dict[str, int]:
        processed_ids = self._processed_game_ids()
        sequence_ids = self._sequence_game_ids()
        pending = sorted(processed_ids - sequence_ids)
        if limit is not None:
            pending = pending[:limit]

        if not pending:
            logger.info('No processed games pending sequence data generation.')
            self.client.close()
            return {'processed_games': 0, 'shot_rows': 0, 'pass_rows': 0}

        processed_games = 0
        shot_rows_total = 0
        pass_rows_total = 0

        for game_id in pending:
            try:
                logger.info('Building sequence data for game_id=%s', game_id)
                game_events = self._load_processed_game_events(game_id)
                if game_events.empty:
                    logger.info('No processed event rows found for game_id=%s', game_id)
                    continue

                shot_sequences = self.build_shot_sequences_for_game(game_events)
                pass_sequences = self.build_passing_sequences_for_game(game_events)
                shot_records = self._records_for_mongo(shot_sequences)
                pass_records = self._records_for_mongo(pass_sequences)

                self.collection_shot_sequences.delete_many({'season': self.config.season.year, 'game_id': game_id})
                self.collection_pass_sequences.delete_many({'season': self.config.season.year, 'game_id': game_id})

                shot_inserted = 0
                pass_inserted = 0
                try:
                    if shot_records:
                        result = self.collection_shot_sequences.insert_many(shot_records, ordered=False)
                        shot_inserted = len(result.inserted_ids)
                    if pass_records:
                        result = self.collection_pass_sequences.insert_many(pass_records, ordered=False)
                        pass_inserted = len(result.inserted_ids)
                except BulkWriteError as bwe:
                    self.collection_shot_sequences.delete_many({'season': self.config.season.year, 'game_id': game_id})
                    self.collection_pass_sequences.delete_many({'season': self.config.season.year, 'game_id': game_id})
                    logger.warning(
                        'BulkWriteError for sequence data game_id=%s, details=%s',
                        game_id,
                        bwe.details,
                    )
                    continue

                processed_games += 1
                shot_rows_total += shot_inserted
                pass_rows_total += pass_inserted
                logger.info(
                    'Inserted sequence data for game_id=%s: shot_rows=%s pass_rows=%s',
                    game_id,
                    shot_inserted,
                    pass_inserted,
                )
            except Exception:
                logger.exception('Error building sequence data for game_id=%s', game_id)

        self.client.close()
        return {
            'processed_games': processed_games,
            'shot_rows': shot_rows_total,
            'pass_rows': pass_rows_total,
        }


def main(config_path: Path | str = CONFIG_PATH) -> Dict[str, int]:
    setup_logging()
    config = load_app_config(str(config_path))
    sequence_data = SequenceData(config)
    return sequence_data.save_sequence_data()


if __name__ == '__main__':
    main()
