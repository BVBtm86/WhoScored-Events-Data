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

    QUALIFYING_OPPONENT_EVENTS = {
        'Pass', 'BallTouch', 'BallRecovery', 'BlockedPass', 'KeeperPickup',
        'Interception', 'Clearance', 'Save', 'TakeOn', 'Tackle', 'Aerial',
        'KeeperSweeper', 'Claim', 'Punch', 'GoodSkill', 'ShieldBallOpp',
    }
    SUBSTITUTION_EVENTS = {'SubstitutionOff', 'SubstitutionOn'}
    PERIOD_BOUNDARY_EVENTS = {
        'Start', 'End', 'EndOfPeriod', 'EndOfGame', 'FullTime',
        'HalfTime', 'PeriodEnd', 'MatchEnd',
    }

    def __init__(self, config: ScrapeDataConfig):
        self.config = config
        self.client = MongoClient(config.mongo.url)
        self.db = self.client[config.mongo.db]
        self.collection_processed_events = self.db[config.mongo.collections['collection_processed_events']]
        self.collection_raw_events = self.db[config.mongo.collections.get('collection_raw_events', 'game_raw_events')]
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

    def _is_shot_event(self, event: pd.Series) -> bool:
        """Return whether a processed event is a shot boundary.

        ``shot_event`` is the canonical processed-event marker and covers
        provider-specific shot type names (including blocked shots).  Keep the
        configured type fallback for callers and older processed documents
        that predate the marker.
        """
        marker = event.get('shot_event')
        if pd.notna(marker):
            return bool(marker)
        return event.get('type') in self.shot_types | {'Goal'}

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

    def _origin_context(self, game_id: int, processed: pd.DataFrame) -> pd.DataFrame:
        """Add raw restart markers omitted by the processed-event projection."""
        raw = list(self.collection_raw_events.find(
            {'season': self.config.season.year, 'game_id': game_id, 'type': {'$in': ['Start', 'Kickoff', 'KickOff', 'KickoffTaken']}},
            {'_id': 0, 'event_idx': 1, 'type': 1, 'period': 1, 'team_id': 1,
             'team': 1, 'outcome_type': 1, 'minute': 1, 'second': 1,
             'x': 1, 'y': 1, 'player_id': 1, 'player': 1},
        ))
        if not raw:
            return processed
        starts = pd.DataFrame(raw)
        return pd.concat([processed, starts], ignore_index=True, sort=False)

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
    def _reconcile_sequence_coordinates(sequences_df: pd.DataFrame) -> pd.DataFrame:
        """Add sequence handoff coordinates while preserving raw event fields."""
        if sequences_df.empty:
            return sequences_df

        result = sequences_df.copy()
        result['sequence_start_x'] = result['x']
        result['sequence_start_y'] = result['y']
        result['sequence_end_x'] = result['end_x']
        result['sequence_end_y'] = result['end_y']
        result['sequence_handoff_gap'] = np.nan
        result['sequence_handoff_reconciled'] = False

        for _, group in result.groupby('sequence_id', sort=False):
            ordered = group.sort_values('sequence_event')
            previous_index = None
            for index in ordered.index:
                if previous_index is not None:
                    previous_end = np.array([
                        result.at[previous_index, 'end_x'],
                        result.at[previous_index, 'end_y'],
                    ], dtype=float)
                    current_start = np.array([
                        result.at[index, 'x'],
                        result.at[index, 'y'],
                    ], dtype=float)
                    if np.isfinite(previous_end).all() and np.isfinite(current_start).all():
                        gap = float(np.linalg.norm(previous_end - current_start))
                        result.at[index, 'sequence_handoff_gap'] = gap
                        # Small gaps are normal coordinate quantisation. Preserve
                        # larger gaps so data discontinuities remain visible.
                        if gap <= 5.0:
                            midpoint = (previous_end + current_start) / 2.0
                            result.at[previous_index, 'sequence_end_x'] = midpoint[0]
                            result.at[previous_index, 'sequence_end_y'] = midpoint[1]
                            result.at[index, 'sequence_start_x'] = midpoint[0]
                            result.at[index, 'sequence_start_y'] = midpoint[1]
                            result.at[index, 'sequence_handoff_reconciled'] = True
                previous_index = index

        return result

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

        if self._is_shot_event(start_event):
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

    @staticmethod
    def _restart_origin(event: Optional[pd.Series], sequence_team_id: Optional[int] = None) -> Optional[str]:
        """Return the restart represented by a recorded pass, if any."""
        if event is None or event.get('type') != 'Pass':
            return None
        if sequence_team_id is not None and event.get('team_id') != sequence_team_id:
            return None
        if event.get('pass_corner') in (1, True):
            return 'Corner kick'
        if event.get('pass_freekick') in (1, True):
            return 'Free kick'
        if event.get('pass_throw_in') in (1, True):
            return 'Throw-in'
        if event.get('pass_goal_kick') in (1, True):
            return 'Goal kick'
        if event.get('pass_kickoff') in (1, True):
            return 'Kickoff'
        if event.get('pass_penalty') in (1, True):
            return 'Penalty kick'
        return None

    @classmethod
    def _restart_type(cls, event: Optional[pd.Series], sequence_team_id: Optional[int] = None) -> Optional[str]:
        """Return a verified restart marker carried by an event.

        A restart is a phase-of-play boundary even when the same team had the
        ball immediately before it. Pass qualifiers are authoritative for the
        restart pass itself; explicit restart event types cover feeds that
        emit the restart separately.
        """
        if event is None:
            return None
        if sequence_team_id is not None and event.get('team_id') != sequence_team_id:
            return None
        qualified = cls._restart_origin(event, sequence_team_id)
        if qualified:
            return qualified
        return {
            'Corner': 'Corner kick',
            'CornerTaken': 'Corner kick',
            'ThrowIn': 'Throw-in',
            'ThrowInTaken': 'Throw-in',
            'FreeKick': 'Free kick',
            'FreeKickTaken': 'Free kick',
            'FreekickTaken': 'Free kick',
            'GoalKick': 'Goal kick',
            'GoalKickTaken': 'Goal kick',
            'Kickoff': 'Kickoff',
            'KickOff': 'Kickoff',
            'KickoffTaken': 'Kickoff',
            'Penalty': 'Penalty kick',
            'PenaltyTaken': 'Penalty kick',
        }.get(event.get('type'))

    @classmethod
    def _origin_event_type(
        cls,
        event: Optional[pd.Series],
        sequence_team_id: int,
        first_pass: Optional[pd.Series] = None,
    ) -> Optional[str]:
        """Classify one candidate without changing sequence boundaries."""
        if event is None:
            return None
        # Restart qualifiers belong to the team that performed the restart.
        # An opposing throw-in/corner/free kick is evidence that the prior
        # possession ended, never the origin of the new team's sequence.
        if event.get('team_id') != sequence_team_id:
            if event.get('type') in {
                'Pass', 'BallTouch', 'BallRecovery', 'BlockedPass',
                'KeeperPickup', 'Interception', 'Clearance', 'Save',
                'TakeOn', 'Tackle', 'Aerial', 'KeeperSweeper', 'Claim',
                'Punch', 'GoodSkill', 'ShieldBallOpp',
            }:
                labels = {
                    'Pass': (
                        'Unsuccessful opponent pass'
                        if event.get('outcome_type') == 'Unsuccessful'
                        else 'Opponent pass'
                    ),
                    'BallTouch': (
                        'Unsuccessful opponent ball touch'
                        if event.get('outcome_type') == 'Unsuccessful'
                        else 'Opponent ball touch'
                    ),
                    'BallRecovery': 'Opponent ball recovery',
                    'BlockedPass': 'Opponent blocked pass',
                    'KeeperPickup': 'Opponent keeper pickup',
                    'Interception': 'Opponent interception',
                    'Clearance': 'Opponent clearance',
                    'Save': 'Opponent save',
                    'TakeOn': 'Opponent take-on',
                    'Tackle': 'Opponent tackle',
                    'Aerial': 'Opponent aerial duel',
                    'KeeperSweeper': 'Opponent keeper sweeper',
                    'Claim': 'Opponent claim',
                    'Punch': 'Opponent punch',
                    'GoodSkill': 'Opponent good skill',
                    'ShieldBallOpp': 'Opponent shield ball',
                }
                return labels[event.get('type')]
            if event.get('type') in {'Turnover', 'Dispossessed', 'Error'}:
                return {
                    'Error': 'Opponent error',
                    'Dispossessed': 'Opponent dispossession',
                    'Turnover': 'Opponent turnover',
                }[event.get('type')]
            return None
        restart = cls._restart_origin(event, sequence_team_id)
        if restart:
            return restart

        event_type = event.get('type')
        outcome = event.get('outcome_type')
        same_team = event.get('team_id') == sequence_team_id

        if event_type in {'Corner', 'CornerTaken'} and same_team:
            return 'Corner kick'
        if event_type in {'ThrowIn', 'ThrowInTaken'} and same_team:
            return 'Throw-in'
        if event_type in {'GoalKick', 'GoalKickTaken'} and same_team:
            return 'Goal kick'
        if event_type in {'Kickoff', 'KickOff'} and same_team:
            return 'Kickoff'

        # A restart pass is allowed to be both origin and first pass.
        if first_pass is event:
            return restart
        if event_type == 'BallRecovery' and same_team:
            return 'Ball recovery'
        if event_type == 'Interception' and same_team:
            return 'Interception'
        if event_type in {'Tackle', 'Challenge'} and same_team and outcome == 'Successful':
            return 'Tackle won'
        if event_type == 'Aerial' and same_team and outcome == 'Successful':
            return 'Aerial duel won'
        if event_type in {'Save', 'Claim', 'Punch', 'Smother', 'KeeperPickup', 'KeeperSweeper'} and same_team:
            return 'Ball recovery'
        if event_type in {'Foul', 'FreeKick', 'FreeKickTaken', 'FreekickTaken'} and same_team and (event_type != 'Foul' or outcome == 'Successful'):
            return 'Free kick'
        if event_type in {'OffsideGiven', 'OffsidePass', 'OffsideProvoked'} and same_team:
            return 'Free kick'
        if same_team and event_type in {'TakeOn', 'BallTouch', 'GoodSkill'} and outcome == 'Successful':
            return {
                'TakeOn': 'Take-on won',
                'BallTouch': 'Ball touch',
                'GoodSkill': 'Good skill',
            }[event_type]
        return None

    @staticmethod
    def _origin_metadata(
        origin: Optional[pd.Series],
        origin_type: Optional[str],
        evidence: str,
    ) -> Dict[str, Any]:
        values = {
            'origin_type': origin_type,
            'origin_event_id': None,
            'origin_event_type': None,
            'origin_player_id': None,
            'origin_player_name': None,
            'origin_team_id': None,
            'origin_minute': None,
            'origin_second': None,
            'origin_x': None,
            'origin_y': None,
            'origin_evidence': evidence,
        }
        if origin is None or not origin_type:
            return values
        values.update({
            'origin_event_id': origin.get('event_idx'),
            'origin_event_type': origin.get('type'),
            'origin_player_id': origin.get('player_id'),
            'origin_player_name': origin.get('player'),
            'origin_team_id': origin.get('team_id'),
            'origin_minute': origin.get('minute'),
            'origin_second': origin.get('second'),
            'origin_x': origin.get('x'),
            'origin_y': origin.get('y'),
        })
        return values

    @classmethod
    def _kickoff_origin_metadata(
        cls,
        event: Optional[pd.Series],
        restarting_team_id: int,
        evidence: str,
        origin_type: str = 'Kickoff',
    ) -> Dict[str, Any]:
        """Build kickoff metadata without borrowing identity from the prior event.

        A post-goal kickoff is often identified from the preceding goal because
        the raw feed does not emit a separate restart row.  The goal scorer is
        therefore not a valid kickoff player, and the goal's team is the team
        that scored rather than the team restarting play.  Keep the recorded
        timestamp/event as evidence, but make the restarting team authoritative
        and only retain a player when the kickoff event itself belongs to that
        team.
        """
        values = cls._origin_metadata(event, origin_type, evidence)
        values['origin_team_id'] = restarting_team_id
        if event is None or event.get('team_id') != restarting_team_id:
            values['origin_player_id'] = None
            values['origin_player_name'] = None
        return values

    @staticmethod
    def _contextual_kickoff_label(
        event: Optional[pd.Series],
        context_df: Optional[pd.DataFrame] = None,
        inferred_post_goal: bool = False,
    ) -> str:
        """Return a kickoff label grounded in period/restart context."""
        if inferred_post_goal:
            return 'Post-goal kickoff'
        period = event.get('period') if event is not None else None
        if period == 'FirstHalf':
            return 'First-half kickoff'
        if period == 'SecondHalf':
            return 'Second-half kickoff'
        normalized = str(period or '').lower().replace(' ', '').replace('_', '')
        if 'extra' in normalized:
            if 'second' in normalized or normalized.endswith('2'):
                return 'Second extra-time kickoff'
            return 'First extra-time kickoff'
        return 'Kickoff'

    @classmethod
    def _contextual_restart_label(
        cls,
        restart: Optional[str],
        event: Optional[pd.Series],
        context_df: Optional[pd.DataFrame] = None,
    ) -> Optional[str]:
        if restart != 'Kickoff':
            return restart
        if event is not None and context_df is not None and not context_df.empty:
            event_idx = event.get('event_idx')
            preceding_goal = any(
                row.get('type') == 'Goal'
                and row.get('event_idx') is not None
                and event_idx is not None
                and row.get('event_idx') < event_idx
                and row.get('period') == event.get('period')
                for _, row in context_df.iterrows()
            )
            if preceding_goal:
                return 'Post-goal kickoff'
        return cls._contextual_kickoff_label(event, context_df)

    @classmethod
    def _classify_sequence_origin(
        cls,
        sequence_df: pd.DataFrame,
        sequence_team_id: int,
        sequence_start_event: Optional[pd.Series],
        context_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Find the latest possession-establishing event before the first pass.

        The origin is contextual metadata only. It is never appended to the
        action rows and therefore cannot alter sequence counts or timing.
        """
        if sequence_df.empty:
            return cls._origin_metadata(None, None, 'empty sequence')
        ordered = sequence_df.copy()
        if 'sequence_event' not in ordered.columns:
            ordered['sequence_event'] = range(1, len(ordered) + 1)
        ordered = ordered.sort_values(['event_idx', 'sequence_event'], kind='stable')
        pass_rows = ordered[(ordered.get('type') == 'Pass') & (ordered.get('outcome_type') == 'Successful')]
        first_pass = pass_rows.iloc[0] if not pass_rows.empty else ordered.iloc[0]

        timeline = context_df.copy() if context_df is not None and not context_df.empty else ordered
        if 'event_idx' not in timeline.columns:
            timeline['event_idx'] = range(len(timeline))
        timeline = timeline.sort_values(['event_idx'], kind='stable')
        restart = cls._restart_origin(first_pass, sequence_team_id)
        if restart:
            return cls._origin_metadata(
                first_pass,
                cls._contextual_restart_label(restart, first_pass, timeline),
                'first pass carries restart qualifier',
            )
        anchor_idx = first_pass.get('event_idx')
        period = first_pass.get('period')
        if pass_rows.empty:
            # For zero-pass buildups, search from the terminal shot rather
            # than excluding the first recorded recovery/action from origin
            # classification. This keeps an explicit recovery as the origin
            # while still allowing a preceding defensive intervention to be
            # referenced when the first row is only an unsuccessful pass.
            terminal = ordered.iloc[-1]
            anchor_idx = terminal.get('event_idx')
            period = terminal.get('period')
        candidates: list[pd.Series] = []
        if sequence_start_event is not None:
            candidates.append(sequence_start_event)
        candidates.extend(
            row for _, row in timeline.iterrows()
            if row.get('period') == period
            and row.get('event_idx') != anchor_idx
            and row.get('event_idx') <= anchor_idx
        )
        origin_context_candidates = list(candidates)

        # Never let origin metadata cross a prior shot or restart. The
        # boundary itself is contextual; only events after it may explain how
        # this sequence began.
        shot_boundaries = [
            row.get('event_idx') for row in candidates
            if row.get('event_idx') is not None
            and row.get('event_idx') < anchor_idx
            and (
                bool(row.get('shot_event'))
                or row.get('type') in {'Goal', 'SavedShot', 'MissedShots', 'ShotOnPost', 'BlockedShot'}
            )
        ]
        restart_boundaries = [
            row.get('event_idx') for row in candidates
            if row.get('event_idx') is not None
            and row.get('event_idx') < anchor_idx
            and cls._restart_type(row, None) is not None
        ]
        latest_shot = max(shot_boundaries) if shot_boundaries else None
        latest_restart = max(restart_boundaries) if restart_boundaries else None
        latest_boundary = max(
            value for value in (latest_shot, latest_restart) if value is not None
        ) if (latest_shot is not None or latest_restart is not None) else None
        if latest_boundary is not None:
            candidates = [
                row for row in candidates
                if row.get('event_idx') is None
                or row.get('event_idx') > latest_boundary
                or (
                    row.get('event_idx') == latest_restart
                    and latest_restart is not None
                    and (latest_shot is None or latest_restart > latest_shot)
                )
            ]

        # A controlled opponent event is the latest evidence boundary. If no
        # more specific origin exists after it, do not inherit an older
        # recovery or turnover from a previous phase.
        controlled_opponent_indices = [
            row.get('event_idx') for row in candidates
            if row.get('event_idx') is not None
            and row.get('event_idx') < anchor_idx
            and cls._controlled_team(row) not in (None, sequence_team_id)
        ]
        if controlled_opponent_indices:
            latest_control = max(controlled_opponent_indices)
            candidates = [
                row for row in candidates
                if row.get('event_idx') is None or row.get('event_idx') >= latest_control
            ]

        # An explicit same-team recovery/keeper recovery as the first action
        # is more specific than an older contextual event.
        if pass_rows.empty and not ordered.empty:
            terminal = ordered.iloc[-1]
            terminal_idx = terminal.get('event_idx')
            prior_shots = [
                row for _, row in timeline.iterrows()
                if row.get('period') == period
                and row.get('event_idx') is not None
                and terminal_idx is not None
                and row.get('event_idx') < terminal_idx
                and row.get('team_id') == sequence_team_id
                and (
                    bool(row.get('shot_event'))
                    or row.get('type') in {'Goal', 'SavedShot', 'MissedShots', 'ShotOnPost', 'BlockedShot'}
                )
            ]
            if prior_shots:
                prior_shot = max(prior_shots, key=lambda row: row.get('event_idx'))
                between = [
                    row for _, row in timeline.iterrows()
                    if row.get('period') == period
                    and row.get('event_idx') is not None
                    and prior_shot.get('event_idx') < row.get('event_idx') < terminal_idx
                ]
                keeper_rebound = any(
                    row.get('team_id') != sequence_team_id
                    and row.get('type') in {'Save', 'Claim', 'Punch', 'Smother', 'KeeperPickup', 'KeeperSweeper'}
                    for row in between
                )
                intervening_non_keeper = [
                    row for row in between
                    if row.get('type') not in {'Save', 'Claim', 'Punch', 'Smother', 'KeeperPickup', 'KeeperSweeper'}
                ]
                if keeper_rebound and not any(
                    cls._is_stoppage_event(row)
                    or cls._restart_type(row, None) is not None
                    or cls._controlled_team(row) not in (None, sequence_team_id)
                    or row.get('type') in {'Pass', 'Clearance', 'BlockedPass', 'Tackle', 'Challenge', 'Interception'}
                    for row in intervening_non_keeper
                ):
                    return cls._origin_metadata(
                        prior_shot,
                        'Shot rebound',
                        f"shot {prior_shot.get('event_idx')} followed by opponent goalkeeper action before next shot",
                    )

            first_action = ordered.iloc[0]
            first_action_origin = cls._origin_event_type(first_action, sequence_team_id, first_action)
            if first_action_origin:
                return cls._origin_metadata(
                    first_action,
                    first_action_origin,
                    f"first included action {first_action.get('event_idx')} classified from recorded type",
                )

        # A Start record immediately before the first controlled pass is the
        # recorded kickoff. The same rule also covers a restart after a goal.
        before = [row for row in origin_context_candidates if row.get('event_idx') is not None]
        for row in reversed(before):
            if row.get('type') == 'Start' and row.get('team_id') == sequence_team_id:
                if anchor_idx is not None and float(anchor_idx) - float(row.get('event_idx')) <= 2:
                    return cls._kickoff_origin_metadata(
                        row,
                        sequence_team_id,
                        'first pass follows recorded period start',
                        cls._contextual_kickoff_label(row, timeline),
                    )
                break
            if row.get('type') in {'Goal', 'SavedShot', 'MissedShots', 'ShotOnPost'}:
                if row.get('type') == 'Goal' and row.get('team_id') != sequence_team_id:
                    goal_idx = row.get('event_idx')
                    explicit_kickoffs = [
                        candidate for _, candidate in timeline.iterrows()
                        if candidate.get('event_idx') is not None
                        and goal_idx is not None
                        and anchor_idx is not None
                        and float(candidate.get('event_idx')) > float(goal_idx)
                        and float(candidate.get('event_idx')) <= float(anchor_idx)
                        and candidate.get('type') in {'Kickoff', 'KickOff', 'KickoffTaken'}
                        and candidate.get('team_id') == sequence_team_id
                    ]
                    explicit_kickoff = explicit_kickoffs[0] if explicit_kickoffs else None
                    if explicit_kickoff is not None:
                        return cls._kickoff_origin_metadata(
                            explicit_kickoff,
                            sequence_team_id,
                            'explicit kickoff follows opponent goal',
                            'Post-goal kickoff',
                        )
                    return cls._kickoff_origin_metadata(
                        None,
                        sequence_team_id,
                        'kickoff inferred after opponent goal; no explicit kickoff event',
                        'Post-goal kickoff',
                    )
                break
        # Prefer a directly preceding unsuccessful opponent pass as the
        # possession origin over the first same-team control touch.  The
        # touch establishes control, but the failed opponent pass explains
        # how the new phase began.
        for candidate in reversed(candidates):
            if (
                candidate.get('type') == 'Pass'
                and candidate.get('team_id') != sequence_team_id
                and candidate.get('outcome_type') == 'Unsuccessful'
                and anchor_idx is not None
                and candidate.get('event_idx') is not None
                and float(anchor_idx) - float(candidate.get('event_idx')) <= 2
            ):
                return cls._origin_metadata(
                    candidate,
                    'Unsuccessful opponent pass',
                    f"event {candidate.get('event_idx')} precedes first controlled action",
                )
        for candidate in reversed(candidates):
            classified = cls._origin_event_type(candidate, sequence_team_id, first_pass)
            if classified == 'Kickoff':
                classified = cls._contextual_restart_label('Kickoff', candidate, timeline)
            if (
                not classified
                and candidate.get('type') == 'Pass'
                and candidate.get('team_id') != sequence_team_id
                and candidate.get('outcome_type') == 'Unsuccessful'
                and anchor_idx is not None
                and float(anchor_idx) - float(candidate.get('event_idx')) <= 2
            ):
                classified = 'Unsuccessful opponent pass'
            if classified:
                return cls._origin_metadata(candidate, classified, f"event {candidate.get('event_idx')} classified from recorded type/qualifiers")

        # A new passing phase can follow a recorded shot without an explicit
        # recovery/restart row. Preserve that traceable hard boundary as the
        # origin instead of leaving the sequence without origin metadata.
        if latest_shot is not None:
            prior_shot = next(
                (row for row in origin_context_candidates if row.get('event_idx') == latest_shot),
                None,
            )
            if prior_shot is not None:
                return cls._origin_metadata(
                    prior_shot,
                    'After shot',
                    f"new phase follows recorded shot event {latest_shot}",
                )

        # A sequence with a same-team possession action immediately before the
        # first pass is supported continuation; otherwise leave it unresolved
        # for diagnostics instead of inventing an origin.
        if pass_rows.empty and any(
            bool(row.get('shot_event'))
            or row.get('type') in {'Goal', 'SavedShot', 'MissedShots', 'ShotOnPost', 'BlockedShot'}
            for _, row in ordered.iterrows()
        ):
            return cls._origin_metadata(
                None,
                'Unknown / Unresolved',
                'shot-only buildup without a verified rebound, recovery, or restart origin',
            )
        return cls._origin_metadata(None, None, 'no possession-establishing or restart event found')

    @classmethod
    def _apply_origin_metadata(
        cls,
        sequence_df: pd.DataFrame,
        sequence_team_id: int,
        sequence_start_event: Optional[pd.Series],
        context_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        metadata = cls._classify_sequence_origin(sequence_df, sequence_team_id, sequence_start_event, context_df)
        if (
            metadata.get('origin_type')
            and not (
                metadata.get('origin_type').lower().startswith(('opponent ', 'unsuccessful opponent '))
                or metadata.get('origin_type') in {'After shot', 'Shot rebound'}
            )
            and metadata.get('origin_team_id') not in (None, sequence_team_id)
        ):
            metadata = cls._origin_metadata(
                None,
                None,
                'origin candidate belonged to the opposing team; origin cleared',
            )
        if metadata.get('origin_type') and metadata.get('origin_player_id') is not None:
            context = context_df if context_df is not None and not context_df.empty else sequence_df
            player_rows = context[context['player_id'] == metadata['origin_player_id']] if 'player_id' in context.columns else pd.DataFrame()
            player_teams = set(player_rows['team_id'].dropna().tolist()) if 'team_id' in player_rows.columns else set()
            if metadata.get('origin_team_id') != sequence_team_id or (
                player_teams and sequence_team_id not in player_teams
            ):
                metadata['origin_player_id'] = None
                metadata['origin_player_name'] = None
                if metadata.get('origin_type') != 'Opponent turnover':
                    metadata['origin_team_id'] = sequence_team_id
        result = sequence_df.copy()
        for key, value in metadata.items():
            result[key] = value
        return result

    @staticmethod
    def _controlled_team(event: pd.Series) -> Optional[int]:
        """Return the team that demonstrably controls the ball in an event."""
        event_type = event.get('type')
        if event.get('outcome_type') != 'Successful':
            return None
        if event_type in {'Pass', 'BallRecovery', 'TakeOn', 'BallTouch', 'GoodSkill'}:
            return event.get('team_id')
        # A successful interception/tackle/aerial is evidence of a challenge
        # outcome, but not by itself proof that the team retained control.
        return None

    @staticmethod
    def _is_stoppage_event(event: pd.Series) -> bool:
        return event.get('type') in {
            'Foul',
            'OffsideGiven',
            'OffsidePass',
            'OffsideProvoked',
        } | SequenceData.SUBSTITUTION_EVENTS | SequenceData.PERIOD_BOUNDARY_EVENTS

    @classmethod
    def _sequence_boundary_decision(
        cls,
        event: pd.Series,
        sequence_team_id: int,
        current_period: Optional[str],
        game_data: Optional[pd.DataFrame] = None,
        event_index: Optional[int] = None,
        direction: str = 'forward',
    ) -> str:
        """Shared hard-boundary policy used by both sequence builders.

        ``break`` is returned for shots, period/stoppage events, restarts and
        qualifying successful opponent actions. Ambiguous unsuccessful
        defensive events remain available to the bounded context resolver.
        """
        if event.get('period') != current_period:
            return 'break'
        if cls._is_shot_event_static(event) or cls._is_stoppage_event(event):
            return 'break'
        if cls._restart_type(event, sequence_team_id) is not None:
            return 'break'
        if event.get('type') in cls.PERIOD_BOUNDARY_EVENTS:
            return 'break'
        if event.get('team_id') != sequence_team_id:
            if (
                event.get('type') in cls.QUALIFYING_OPPONENT_EVENTS
                and event.get('outcome_type') == 'Successful'
            ):
                return 'break'
        return 'continue'

    @staticmethod
    def _is_shot_event_static(event: pd.Series) -> bool:
        marker = event.get('shot_event')
        if pd.notna(marker):
            return bool(marker)
        return event.get('type') in {'Goal', 'SavedShot', 'MissedShots', 'ShotOnPost', 'BlockedShot'}

    def _bounded_control_evidence(
        self,
        game_data: pd.DataFrame,
        event_index: int,
        sequence_team_id: int,
        current_period: Optional[str],
        lookahead: int = 6,
        same_team_shot_continues: bool = True,
    ) -> Optional[str]:
        """Resolve a defensive intervention using a bounded chronological window.

        The caller examines events after the intervention, whether it is scanning
        forward for a passing chain or backward from a shot.  A controlled action
        is stronger evidence than a generic successful defensive event.
        """
        end = min(len(game_data), event_index + 1 + lookahead)
        for index in range(event_index + 1, end):
            candidate = game_data.iloc[index]
            if candidate.get('period') != current_period:
                return 'stoppage'
            if self._is_stoppage_event(candidate):
                return 'stoppage'
            if self._is_shot_event(candidate):
                # A shot by the sequence team is evidence that the team
                # retained control through the intervention. It is terminal
                # for the buildup, but not a possession boundary before it.
                if candidate.get('team_id') == sequence_team_id:
                    return 'same_team' if same_team_shot_continues else 'stoppage'
                return 'stoppage'
            controlled_team = self._controlled_team(candidate)
            if controlled_team == sequence_team_id:
                return 'same_team'
            if controlled_team is not None and controlled_team != sequence_team_id:
                return 'opponent'
            # Restart markers are boundaries even when the raw feed does not
            # provide a player or a successful outcome for the restart.
            if candidate.get('type') in {'Start', 'Kickoff', 'KickOff', 'KickoffTaken'}:
                return 'stoppage'
        return None

    def _intervention_decision(
        self,
        event: pd.Series,
        sequence_team_id: int,
        game_data: Optional[pd.DataFrame],
        event_index: Optional[int],
        current_period: Optional[str],
        same_team_shot_continues: bool = True,
    ) -> str:
        """Return keep/ignore/break for an opponent defensive intervention."""
        if game_data is None or event_index is None:
            return 'ignore' if event.get('outcome_type') == 'Unsuccessful' else 'break'
        evidence = self._bounded_control_evidence(
            game_data,
            event_index,
            sequence_team_id,
            current_period,
            same_team_shot_continues=same_team_shot_continues,
        )
        if evidence == 'same_team':
            return 'ignore'
        if evidence in {'opponent', 'stoppage'}:
            return 'break'
        # An unsuccessful defensive action is context only. A successful but
        # unresolved intervention is conservatively treated as a boundary.
        return 'ignore' if event.get('outcome_type') == 'Unsuccessful' else 'break'

    def _shooting_valid_sequence_event(
        self,
        event: pd.Series,
        shooting_team_id: int,
        current_period: str,
        game_data: Optional[pd.DataFrame] = None,
        event_index: Optional[int] = None,
    ) -> str:
        event_type = event['type']
        outcome = event.get('outcome_type')
        event_period = event.get('period')

        shared_boundary = self._sequence_boundary_decision(
            event, shooting_team_id, current_period, game_data, event_index, direction='backward'
        )
        if shared_boundary == 'break':
            # A restart pass is part of the new buildup, but it is also the
            # hard boundary that stops the backward scan.
            if event.get('team_id') == shooting_team_id and event.get('type') == 'Pass' and self._restart_type(event, shooting_team_id):
                return 'keep_break'
            return 'break'

        if event_type in self.ignore_types:
            return 'ignore'

        if event_type in self.break_types:
            return 'break'

        same_team = event['team_id'] == shooting_team_id

        if not same_team:
            if event_type == 'Error':
                return self._intervention_decision(
                    event,
                    shooting_team_id,
                    game_data,
                    event_index,
                    current_period,
                )
            if event_type in self.opponent_pressure_types or event_type in {
                'BlockedPass',
                'Clearance',
                'Interception',
                'Tackle',
                'Challenge',
            }:
                return self._intervention_decision(
                    event,
                    shooting_team_id,
                    game_data,
                    event_index,
                    current_period,
                )
            return 'break'

        if event_type in {'Dispossessed', 'Turnover', 'Error'}:
            evidence = self._bounded_control_evidence(
                game_data,
                event_index,
                shooting_team_id,
                current_period,
            ) if game_data is not None and event_index is not None else None
            return 'keep' if evidence == 'same_team' else 'break'

        # A prior shot is always a display-chain boundary. A rebound may let
        # the same team shoot again, but the new buildup must not absorb the
        # passes that preceded the earlier shot.
        if self._is_shot_event(event):
            return 'break'

        if event_type in {'Pass', 'TakeOn', 'BallTouch'}:
            if outcome == 'Successful':
                return 'keep'
            evidence = self._bounded_control_evidence(
                game_data,
                event_index,
                shooting_team_id,
                current_period,
            ) if game_data is not None and event_index is not None else None
            return 'keep' if evidence == 'same_team' else 'break'

        if event_type in self.sequence_action_types:
            return 'break' if outcome == 'Unsuccessful' else 'keep'

        return 'ignore'

    def _get_shot_sequence(
        self,
        game_data: pd.DataFrame,
        shot_index: int,
        sequence_id: int,
        origin_context: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        raw_shot_team_id = game_data.loc[shot_index, 'team_id']
        own_goal = bool(game_data.loc[shot_index].get('shot_own_goal', False))
        shooting_team_id = raw_shot_team_id
        if own_goal:
            # WhoScored assigns an own-goal shot to the defending player. The
            # buildup, however, belongs to the attacking team that supplied
            # the preceding pass/cross. Use the nearest preceding pass in the
            # same period as the possession owner, while retaining the own
            # goal row as the terminal shot.
            shot_period = game_data.loc[shot_index, 'period']
            for prior_index in range(shot_index - 1, -1, -1):
                prior = game_data.loc[prior_index]
                if prior.get('period') != shot_period:
                    break
                if prior.get('type') == 'Pass' and prior.get('team_id') != raw_shot_team_id:
                    shooting_team_id = prior.get('team_id')
                    break
        current_period = game_data.loc[shot_index, 'period']
        sequence_rows = [game_data.loc[shot_index]]
        start_event = None
        j = shot_index - 1

        while j >= 0:
            event = game_data.loc[j]
            # A cross that leads directly to an own goal can be recorded as an
            # unsuccessful pass because the defender touched it. Keep that
            # terminal attacking pass as part of the buildup; normal shot
            # sequences still use the regular possession rules.
            if own_goal and j == shot_index - 1 and event.get('type') == 'Pass' and event.get('team_id') == shooting_team_id:
                sequence_rows.append(event)
                j -= 1
                continue
            decision = self._shooting_valid_sequence_event(
                event,
                shooting_team_id,
                current_period,
                game_data,
                j,
            )

            if decision == 'ignore':
                j -= 1
                continue

            if decision == 'keep':
                sequence_rows.append(event)
                j -= 1
                continue

            if decision == 'keep_break':
                sequence_rows.append(event)
                start_event = event
                break

            if decision == 'break':
                start_event = event
                break

        sequence_df = pd.DataFrame(sequence_rows).sort_index()
        sequence_df['sequence_id'] = sequence_id
        sequence_df['sequence_key'] = str(game_data.loc[shot_index, 'game_id']) + '_' + str(sequence_id)
        sequence_df['sequence_event'] = range(1, len(sequence_df) + 1)
        sequence_df['shot_event_idx'] = game_data.loc[shot_index, 'event_idx']
        # Keep the sequence owner explicit. The first chronological row may be
        # a recovery/pass after an opponent action, so deriving ownership from
        # sequence_df.iloc[0].team_id is unsafe.
        sequence_df['sequence_team_id'] = shooting_team_id
        owner_rows = game_data[game_data['team_id'] == shooting_team_id]
        sequence_df['sequence_team'] = owner_rows.iloc[-1].get('team') if not owner_rows.empty else game_data.loc[shot_index, 'team']
        # Sequence-start metadata describes the first qualifying action in the
        # stored sequence.  The preceding boundary event belongs exclusively
        # to origin classification and must never be persisted as the start of
        # the sequence (for example, a tackle or recovery before the first
        # pass).
        qualifying_passes = sequence_df[
            (sequence_df['type'] == 'Pass')
            & (sequence_df['outcome_type'] == 'Successful')
        ]
        first_action = qualifying_passes.iloc[0] if not qualifying_passes.empty else sequence_df.iloc[0]
        sequence_df['sequence_start_reason'] = (
            'First qualifying pass' if not qualifying_passes.empty else 'Shot without prior successful pass'
        )
        sequence_df = sequence_df.assign(**self._event_metadata(first_action, 'sequence_start'))
        sequence_df = sequence_df.assign(**self._event_metadata(game_data.loc[shot_index], 'sequence_end'))
        sequence_df = self._apply_origin_metadata(sequence_df, shooting_team_id, start_event, origin_context if origin_context is not None else game_data)
        sequence_df['sequence_end_side'] = 'same_team'
        sequence_df['sequence_ending'] = 'Shot'

        return sequence_df

    @staticmethod
    def _has_player_connected_shot_buildup(sequence_df: pd.DataFrame) -> bool:
        """Accept only shot buildups with a real teammate exchange.

        A take-on followed directly by a shot is a valid football action, but
        it is not a player-to-player buildup and must not be presented as one.
        """
        if sequence_df.empty or 'type' not in sequence_df or 'team_id' not in sequence_df:
            return False
        passes = sequence_df[
            (sequence_df['type'] == 'Pass')
            & (sequence_df['outcome_type'] == 'Successful')
        ]
        players = sequence_df['player_id'].dropna().unique()
        return not passes.empty and len(players) >= 2

    def build_shot_sequences_for_game(self, game_data: pd.DataFrame, origin_context: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if game_data.empty:
            return pd.DataFrame()

        game_data = game_data.sort_values('event_idx').reset_index(drop=True)
        sequences = []
        sequence_id = 1

        shot_mask = game_data['shot_event'].fillna(False).astype(bool)
        shot_indexes = game_data[shot_mask].index

        for shot_index in shot_indexes:
            sequence_df = self._get_shot_sequence(game_data, shot_index, sequence_id, origin_context)
            if not sequence_df.empty:
                sequences.append(sequence_df)
                sequence_id += 1

        if not sequences:
            return pd.DataFrame()

        sequences_df = pd.concat(sequences, ignore_index=True)
        sequences_df = self._reconcile_sequence_coordinates(sequences_df)
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
        game_data: Optional[pd.DataFrame] = None,
        event_index: Optional[int] = None,
    ) -> str:
        event_type = event['type']
        outcome = event.get('outcome_type')
        event_period = event.get('period')
        same_team = event['team_id'] == passing_team_id

        if self._sequence_boundary_decision(
            event, passing_team_id, current_period, game_data, event_index, direction='forward'
        ) == 'break':
            return 'break'

        if event_type in self.ignore_types and event_type not in self.SUBSTITUTION_EVENTS:
            return 'ignore'

        if event_type in {'OffsideGiven', 'OffsideProvoked'}:
            return 'ignore'

        if event_type == 'OffsidePass':
            return 'keep_break' if same_team else 'break'

        if not same_team:
            if event_type == 'Error':
                return 'ignore'
            if event_type in self.opponent_pressure_types or event_type in {
                'BlockedPass',
                'Clearance',
                'Interception',
                'Tackle',
                'Challenge',
            }:
                return self._intervention_decision(
                    event,
                    passing_team_id,
                    game_data,
                    event_index,
                    current_period,
                    same_team_shot_continues=False,
                )
            return 'keep_break'

        if event_type == 'Error':
            return 'keep'

        if event_type in {'Dispossessed', 'Turnover'}:
            evidence = self._bounded_control_evidence(
                game_data,
                event_index,
                passing_team_id,
                current_period,
                same_team_shot_continues=False,
            ) if game_data is not None and event_index is not None else None
            return 'ignore' if evidence == 'same_team' else 'keep_break'

        if event_type in self.break_types:
            return 'keep_break'

        if event_type in {'Pass', 'BallTouch', 'TakeOn', 'GoodSkill'} and outcome == 'Successful':
            return 'keep'

        if event_type == 'Pass' and outcome != 'Successful':
            evidence = self._bounded_control_evidence(
                game_data,
                event_index,
                passing_team_id,
                current_period,
            ) if game_data is not None and event_index is not None else None
            return 'keep' if evidence == 'same_team' else 'keep_break'

        if event_type == 'BallTouch' and outcome != 'Successful':
            evidence = self._bounded_control_evidence(
                game_data,
                event_index,
                passing_team_id,
                current_period,
            ) if game_data is not None and event_index is not None else None
            return 'ignore' if evidence == 'same_team' else 'keep_break'

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

        if same_team and self._is_shot_event(event):
            return 'Shot'

        if same_team and event_type == 'Pass' and outcome != 'Successful':
            return 'Unsuccessful Pass'

        if same_team and event_type == 'BallTouch' and outcome != 'Successful':
            return 'Unsuccessful Ball Touch'

        if same_team and event_type in {'OffsideGiven', 'OffsidePass', 'OffsideProvoked'}:
            return 'Offside'

        if not same_team:
            if event_type == 'Pass':
                return 'Opponent pass' if outcome == 'Successful' else 'Unsuccessful opponent pass'
            if event_type == 'OffsidePass':
                return 'Opponent offside pass'
            return event_type

        return event_type

    def _add_passing_sequence_end_metadata(
        self,
        current_rows: list[pd.Series],
        start_event: Optional[pd.Series],
        current_team_id: int,
        event: pd.Series,
        game_data: pd.DataFrame,
        sequence_ending: str,
        origin_context: Optional[pd.DataFrame] = None,
        period_changed: bool = False,
    ) -> list[pd.Series]:
        sequence_team = current_rows[0].get('team')
        # Keep the terminating event separate from the last displayed pass.
        # The latter is the final row in current_rows; sequence_end_* must
        # identify the event that actually ended the chain, including an
        # opponent pass/recovery when possession changed.
        sequence_end_event = event

        for row in current_rows:
            first_action = current_rows[0]
            row['sequence_start_reason'] = 'First qualifying pass'
            for key, value in SequenceData._event_metadata(first_action, 'sequence_start').items():
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
                    'same_team' if event.get('team_id') == current_team_id else 'opponent'
                )

        origin_metadata = self._classify_sequence_origin(
            pd.DataFrame(current_rows), current_team_id, start_event,
            origin_context if origin_context is not None else game_data
        )
        for row in current_rows:
            for key, value in origin_metadata.items():
                row[key] = value

        return current_rows

    def build_passing_sequences_for_game(self, game_data: pd.DataFrame, origin_context: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if game_data.empty:
            return pd.DataFrame()

        # event_idx is the provider's canonical chronological order. Sorting
        # by the human-readable period label can place extra periods or
        # stoppage rows incorrectly, so never use the label as the primary key.
        game_data = game_data.sort_values('event_idx').reset_index(drop=True)

        sequences = []
        current_rows: list[pd.Series] = []
        current_team_id = None
        current_period = None
        current_start_event = None
        sequence_id = 1

        for idx, event in game_data.iterrows():
            event_type = event['type']
            outcome = event.get('outcome_type')

            restart = self._restart_type(event, current_team_id)
            if current_team_id is not None and restart:
                # A restart begins a new phase even if the restarting team is
                # unchanged. Finalize the prior chain before considering this
                # event as the first pass of the new chain.
                sequence_ending = restart
                current_rows = self._add_passing_sequence_end_metadata(
                    current_rows=current_rows,
                    start_event=current_start_event,
                    current_team_id=current_team_id,
                    event=event,
                    game_data=game_data,
                    sequence_ending=sequence_ending,
                    origin_context=origin_context,
                )
                sequences.extend(current_rows)
                sequence_id += 1
                current_rows = []
                current_team_id = None
                current_period = None
                current_start_event = None

            if restart and event_type != 'Pass':
                continue

            if current_team_id is None:
                if event_type == 'Pass' and outcome == 'Successful':
                    current_team_id = event['team_id']
                    current_period = event['period']
                    current_start_event = game_data.loc[idx - 1] if idx > 0 else None

                    row = event.copy()
                    row['sequence_id'] = sequence_id
                    current_rows = [row]
                continue

            pass_decision = self._passing_valid_sequence_event(
                event,
                current_team_id,
                current_period,
                game_data,
                idx,
            )

            if pass_decision == 'keep':
                # Non-pass controlled actions keep possession alive but are
                # deliberately not stored as displayed passing actions.
                if event_type == 'Pass':
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
                    game_data=game_data,
                    sequence_ending=sequence_ending,
                    origin_context=origin_context,
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
                    game_data=game_data,
                    sequence_ending=sequence_ending,
                    origin_context=origin_context,
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
                row['sequence_start_reason'] = 'First qualifying pass'
                for key, value in self._event_metadata(current_rows[0], 'sequence_start').items():
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

            origin_metadata = self._classify_sequence_origin(
                pd.DataFrame(current_rows), current_team_id, current_start_event,
                origin_context if origin_context is not None else game_data
            )
            for row in current_rows:
                for key, value in origin_metadata.items():
                    row[key] = value

            sequences.extend(current_rows)

        sequences_df = pd.DataFrame(sequences)
        if sequences_df.empty:
            return sequences_df

        sequences_df['sequence_event'] = sequences_df.groupby('sequence_id').cumcount() + 1
        sequences_df['sequence_key'] = (
            sequences_df['game_id'].astype(str) + '_' + sequences_df['sequence_id'].astype(str)
        )
        sequences_df = sequences_df.groupby('sequence_id', group_keys=False).filter(
            lambda group: int(((group['type'] == 'Pass') & (group['outcome_type'] == 'Successful')).sum()) >= 2
        )
        # Shot-ending chains are represented by the shot-buildup collection;
        # keep the passing collection strictly for non-shot passing sequences.
        if not sequences_df.empty and 'sequence_ending' in sequences_df.columns:
            sequences_df = sequences_df[~sequences_df['sequence_ending'].isin({'Shot', 'Goal'})]
        if sequences_df.empty:
            return sequences_df
        sequences_df = self._reconcile_sequence_coordinates(sequences_df)

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

                origin_context = self._origin_context(game_id, game_events)
                shot_sequences = self.build_shot_sequences_for_game(game_events, origin_context)
                pass_sequences = self.build_passing_sequences_for_game(game_events, origin_context)
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
