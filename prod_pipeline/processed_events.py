from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Set
import ast

import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

from helper import setup_logging, load_app_config, logger
from config import ScrapeDataConfig

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'


class ProcessedEvents:
    def __init__(self, config: ScrapeDataConfig):
        self.config = config
        self.client = MongoClient(config.mongo.url)
        self.db = self.client[config.mongo.db]
        self.collection_raw_events = self.db[config.mongo.collections['collection_raw_events']]
        self.collection_processed_events = self.db[config.mongo.collections['collection_processed_events']]

    def _raw_game_ids(self) -> Set[int]:
        return {
            int(x)
            for x in self.collection_raw_events.distinct('game_id', {'season': self.config.season.year})
            if x is not None
        }

    def _processed_game_ids(self) -> Set[int]:
        return {
            int(x)
            for x in self.collection_processed_events.distinct('game_id', {'season': self.config.season.year})
            if x is not None
        }

    @staticmethod
    def _coerce_qualifiers(value: Any) -> list[dict]:
        if value is None:
            return []
        if isinstance(value, float) and pd.isna(value):
            return []
        if isinstance(value, list):
            return [q for q in value if isinstance(q, dict)]
        if isinstance(value, dict):
            return [value]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                return []
            return ProcessedEvents._coerce_qualifiers(parsed)
        return []

    @staticmethod
    def _build_qset(qualifiers: Any) -> set[str]:
        names = set()
        for q in ProcessedEvents._coerce_qualifiers(qualifiers):
            qtype = q.get('type') or {}
            display_name = qtype.get('displayName') if isinstance(qtype, dict) else None
            if display_name:
                names.add(display_name)
        return names

    @staticmethod
    def _build_qmap(qualifiers: Any) -> dict[str, Any]:
        qmap: dict[str, Any] = {}
        for q in ProcessedEvents._coerce_qualifiers(qualifiers):
            qtype = q.get('type') or {}
            display_name = qtype.get('displayName') if isinstance(qtype, dict) else None
            if display_name:
                qmap[display_name] = q.get('value')
        return qmap

    @staticmethod
    def _ensure_qual_cols(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'qualifiers' in df.columns:
            df['qset'] = df['qualifiers'].apply(ProcessedEvents._build_qset)
            df['qmap'] = df['qualifiers'].apply(ProcessedEvents._build_qmap)
        elif 'qualifier_names' in df.columns:
            df['qset'] = df['qualifier_names'].apply(
                lambda x: set(x) if isinstance(x, list) else set()
            )
            df['qmap'] = [{} for _ in range(len(df))]
        else:
            df['qset'] = [set() for _ in range(len(df))]
            df['qmap'] = [{} for _ in range(len(df))]
        return df

    @staticmethod
    def _has(df: pd.DataFrame, qualifier: str) -> pd.Series:
        return df['qset'].apply(lambda values: qualifier in values)

    @staticmethod
    def _has_any(df: pd.DataFrame, qualifiers: set[str]) -> pd.Series:
        return df['qset'].apply(lambda values: bool(values & qualifiers))

    @staticmethod
    def _qvalue(qmap: dict[str, Any], key: str) -> Any:
        return qmap.get(key)

    @staticmethod
    def _num(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors='coerce')

    @staticmethod
    def _normalize_mongo_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return {k: ProcessedEvents._normalize_mongo_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [ProcessedEvents._normalize_mongo_value(v) for v in value]
        if isinstance(value, tuple):
            return [ProcessedEvents._normalize_mongo_value(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
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

    @classmethod
    def build_processed_game_events(cls, raw_events: pd.DataFrame) -> pd.DataFrame:
        events = cls._ensure_qual_cols(raw_events).copy()
        events['type'] = events['type'].fillna('')
        events['outcome_type'] = events['outcome_type'].fillna('')

        qmap = events['qmap']
        t = events['type']
        outcome = events['outcome_type']
        has = cls._has
        has_any = cls._has_any

        pass_mask = t.eq('Pass')
        shot_mask = t.isin({'Goal', 'MissedShots', 'SavedShot', 'ShotOnPost'})

        small_box_quals = {'SmallBoxCentre', 'SmallBoxLeft', 'SmallBoxRight', 'SixYardBox'}
        box_quals = {'BoxCentre', 'BoxLeft', 'BoxRight', 'DeepBoxCentre', 'DeepBoxLeft', 'DeepBoxRight', 'PenaltyArea'}
        out_box_quals = {
            'OutOfBoxCentre', 'OutOfBoxLeft', 'OutOfBoxRight',
            'OutOfBoxDeepCentre', 'OutOfBoxDeepLeft', 'OutOfBoxDeepRight',
            'ThirtyFivePlusCentre', 'ThirtyFivePlusLeft', 'ThirtyFivePlusRight',
        }
        set_piece_quals = {
            'DirectFreeKick', 'DirectFreekick', 'FromCorner', 'CornerTaken',
            'SetPiece', 'ThrowIn', 'ThrowinSetPiece', 'ThrowInSetPiece',
            'IndirectFreeKickTaken', 'IndirectFreekickTaken', 'IndirectFreekick', 'FreekickTaken',
            'FreeKickTaken',
        }

        own_goal = t.eq('Goal') & has(events, 'OwnGoal')
        shot_blocked = t.eq('SavedShot') & has(events, 'Blocked')
        shot_woodwork = t.eq('ShotOnPost') & ~own_goal
        shot_goal = t.eq('Goal') & ~own_goal
        shot_on_target = t.isin({'Goal', 'SavedShot'}) & ~shot_blocked & ~own_goal
        shot_off_target = (t.eq('MissedShots') | shot_woodwork) & ~shot_blocked & ~own_goal

        events['shot_result'] = np.select(
            [own_goal, shot_goal, shot_on_target, shot_off_target, shot_blocked, shot_woodwork],
            ['Own Goal', 'Goal', 'On Target', 'Off Target', 'Blocked', 'Woodwork'],
            default=None,
        )
        events['shot_zone'] = np.select(
            [has_any(events, small_box_quals), has_any(events, box_quals), has_any(events, out_box_quals)],
            ['6-yard box', 'Penalty Area', 'Outside of box'],
            default='Unknown',
        )
        events.loc[~shot_mask, 'shot_zone'] = None
        shot_penalty = shot_mask & has(events, 'Penalty') & ~own_goal
        shot_fastbreak = shot_mask & has(events, 'FastBreak') & ~own_goal
        shot_set_piece = shot_mask & has_any(events, set_piece_quals) & ~shot_penalty & ~own_goal
        shot_open_play = shot_mask & has(events, 'RegularPlay') & ~shot_penalty & ~shot_fastbreak & ~shot_set_piece & ~own_goal
        events['shot_situation'] = np.select(
            [own_goal, shot_penalty, shot_fastbreak, shot_set_piece, shot_open_play],
            ['Own Goal', 'Penalty', 'Fastbreak', 'Set Pieces', 'Open Play'],
            default='Unknown',
        )
        events.loc[~shot_mask, 'shot_situation'] = None
        events['shot_body_part'] = np.select(
            [has(events, 'RightFoot'), has(events, 'LeftFoot'), has(events, 'Head')],
            ['Right foot', 'Left foot', 'Head'],
            default='Other body parts',
        )
        events.loc[~shot_mask, 'shot_body_part'] = None
        events['shot_is_goal'] = shot_goal.astype(int)
        events['shot_on_target'] = shot_on_target.astype(int)
        events['shot_off_target'] = shot_off_target.astype(int)
        events['shot_blocked'] = shot_blocked.astype(int)
        events['shot_woodwork'] = shot_woodwork.astype(int)
        events['shot_own_goal'] = own_goal.astype(int)
        related_player_id = pd.to_numeric(events.get('related_player_id'), errors='coerce')
        events['goal_assist'] = (shot_goal & has(events, 'Assisted') & related_player_id.notna()).astype(int)

        px = cls._num(events['x'])
        py = cls._num(events['y'])
        pex = cls._num(events['end_x'])
        pey = cls._num(events['end_y'])
        pdx = pex - px
        pdy = pey - py
        target_zone_1 = 100 / 3
        target_zone_2 = 200 / 3
        freekick_quals = {'FreekickTaken', 'FreeKickTaken', 'IndirectFreekickTaken', 'IndirectFreeKickTaken'}
        long_excluded_type = has_any(events, {'Cross', 'KeyPass', 'Throughball'})
        pass_long = pass_mask & has(events, 'Longball') & ~long_excluded_type

        events['pass_successful'] = (pass_mask & outcome.eq('Successful')).astype(int)
        events['pass_length'] = np.where(pass_long, 'Long', 'Short')
        events.loc[~pass_mask, 'pass_length'] = None
        events['pass_height'] = np.where(has(events, 'Chipped'), 'Chipped', 'Ground')
        events.loc[~pass_mask, 'pass_height'] = None
        events['pass_body_part'] = np.where(has(events, 'HeadPass'), 'Head', 'Feet')
        events.loc[~pass_mask, 'pass_body_part'] = None
        events['pass_target_zone'] = np.select(
            [pex < target_zone_1, (pex >= target_zone_1) & (pex < target_zone_2), pex >= target_zone_2],
            ['Defensive Third', 'Mid Third', 'Final Third'],
            default=None,
        )
        events.loc[~pass_mask, 'pass_target_zone'] = None
        events['pass_forward'] = (pass_mask & pdx.gt(0)).astype(int)
        events['pass_backward'] = (pass_mask & pdx.lt(0)).astype(int)
        events['pass_left'] = (pass_mask & pdy.gt(0)).astype(int)
        events['pass_right'] = (pass_mask & pdy.lt(0)).astype(int)
        events['pass_cross'] = (pass_mask & has(events, 'Cross')).astype(int)
        events['pass_freekick'] = (pass_mask & has_any(events, freekick_quals) & ~has(events, 'Cross')).astype(int)
        events['pass_corner'] = (pass_mask & has(events, 'CornerTaken')).astype(int)
        events['pass_through_ball'] = (pass_mask & has(events, 'Throughball')).astype(int)
        events['pass_throw_in'] = (pass_mask & has(events, 'ThrowIn')).astype(int)
        events['pass_key_pass_qualifier'] = (pass_mask & has(events, 'KeyPass')).astype(int)
        events['shot_assisted_key_pass'] = (shot_mask & has(events, 'Assisted')).astype(int)

        events['tackle_result'] = np.select(
            [t.eq('Tackle') & outcome.eq('Successful'), t.eq('Tackle') & outcome.ne('Successful'), t.eq('Challenge')],
            ['Gained Possession', 'Did Not Get Possession', 'Was Dribbled'],
            default=None,
        )
        outfielder_block = t.eq('Save') & has(events, 'OutfielderBlock')
        blocked_cross = t.eq('BlockedPass') | (t.eq('Clearance') & has(events, 'BlockedCross'))
        blocked_shot = t.eq('Block') | outfielder_block
        block_mask = blocked_cross | blocked_shot
        clearance_mask = t.eq('Clearance') & ~has(events, 'BlockedCross')
        events['clearance_body_part'] = np.select(
            [clearance_mask & has(events, 'Head'), clearance_mask],
            ['Head', 'Feet'],
            default=None,
        )
        events['block_type'] = np.select(
            [blocked_shot, blocked_cross],
            ['Blocked Shot', 'Blocked Cross'],
            default=None,
        )
        events['offside_type'] = np.select(
            [t.eq('OffsideGiven'), t.eq('OffsidePass'), t.eq('OffsideProvoked')],
            ['Caught Offside', 'Offside Pass', 'Offside Provoked'],
            default=None,
        )
        events['foul_type'] = np.select(
            [
                t.eq('Foul') & outcome.eq('Unsuccessful'),
                t.eq('Foul') & outcome.eq('Successful'),
            ],
            ['Foul Committed', 'Foul Suffered'],
            default=None,
        )
        events['aerial_result'] = np.select(
            [t.eq('Aerial') & outcome.eq('Successful'), t.eq('Aerial') & outcome.ne('Successful')],
            ['Won', 'Lost'],
            default=None,
        )
        dispossessed_mask = t.eq('Dispossessed')
        turnover_mask = t.eq('BallTouch') & outcome.eq('Unsuccessful')
        events['loss_possession_type'] = np.select(
            [dispossessed_mask, turnover_mask],
            ['Dispossessed', 'Turnover'],
            default=None,
        )
        error_mask = has_any(events, {'LeadingToGoal', 'LeadingToAttempt'})
        events['error_type'] = np.select(
            [has(events, 'LeadingToGoal'), has(events, 'LeadingToAttempt')],
            ['Error Leading to Goal', 'Error Leading to Shot'],
            default=None,
        )
        events['gk_type'] = np.select(
            [t.eq('Save') & ~has(events, 'OutfielderBlock'), t.eq('Claim'), t.eq('Punch'), t.eq('KeeperPickup'), t.eq('KeeperSweeper')],
            ['Save', 'Claim', 'Punch', 'Keeper Pickup', 'Keeper Sweeper'],
            default=None,
        )
        events['card_type'] = np.select(
            [has(events, 'SecondYellow'), has(events, 'Yellow'), has(events, 'Red')],
            ['Second Yellow', 'Yellow', 'Red'],
            default=None,
        )
        events['substitution_type'] = np.select(
            [t.eq('SubstitutionOff'), t.eq('SubstitutionOn')],
            ['Off', 'On'],
            default=None,
        )

        events['is_shot'] = shot_mask.astype(int)
        events['is_pass'] = pass_mask.astype(int)
        events['is_pass_attempt'] = pass_mask.astype(int)
        events['is_pass_completed'] = events['pass_successful']
        events['is_pass_incomplete'] = (pass_mask & ~outcome.eq('Successful')).astype(int)
        events['is_pass_long'] = pass_long.astype(int)
        events['is_pass_short'] = (pass_mask & ~pass_long).astype(int)
        events['is_pass_chipped'] = (pass_mask & has(events, 'Chipped')).astype(int)
        events['is_pass_ground'] = (pass_mask & ~has(events, 'Chipped')).astype(int)
        events['is_pass_head'] = (pass_mask & has(events, 'HeadPass')).astype(int)
        events['is_pass_feet'] = (pass_mask & ~has(events, 'HeadPass')).astype(int)
        events['is_pass_forward'] = events['pass_forward']
        events['is_pass_backward'] = events['pass_backward']
        events['is_pass_left'] = events['pass_left']
        events['is_pass_right'] = events['pass_right']
        events['is_pass_defensive_third'] = (pass_mask & events['pass_target_zone'].eq('Defensive Third')).astype(int)
        events['is_pass_mid_third'] = (pass_mask & events['pass_target_zone'].eq('Mid Third')).astype(int)
        events['is_pass_final_third'] = (pass_mask & events['pass_target_zone'].eq('Final Third')).astype(int)
        events['is_pass_cross'] = events['pass_cross']
        events['is_pass_freekick'] = events['pass_freekick']
        events['is_pass_corner'] = events['pass_corner']
        events['is_pass_through_ball'] = events['pass_through_ball']
        events['is_pass_throw_in'] = events['pass_throw_in']
        events['is_pass_key_pass_qualifier'] = events['pass_key_pass_qualifier']
        events['is_key_pass'] = events['shot_assisted_key_pass']
        events['is_dribble'] = t.eq('TakeOn').astype(int)
        events['is_dribble_successful'] = (t.eq('TakeOn') & outcome.eq('Successful')).astype(int)
        events['is_dribble_unsuccessful'] = (t.eq('TakeOn') & outcome.ne('Successful')).astype(int)
        events['is_ball_recovery'] = t.eq('BallRecovery').astype(int)
        events['is_tackle'] = t.isin({'Tackle', 'Challenge'}).astype(int)
        events['is_tackle_gained_possession'] = events['tackle_result'].eq('Gained Possession').astype(int)
        events['is_tackle_did_not_get_possession'] = events['tackle_result'].eq('Did Not Get Possession').astype(int)
        events['is_tackle_was_dribbled'] = events['tackle_result'].eq('Was Dribbled').astype(int)
        events['is_interception'] = t.eq('Interception').astype(int)
        events['is_clearance'] = clearance_mask.astype(int)
        events['is_clearance_head'] = events['clearance_body_part'].eq('Head').astype(int)
        events['is_clearance_feet'] = events['clearance_body_part'].eq('Feet').astype(int)
        events['is_block'] = block_mask.astype(int)
        events['is_blocked_shot'] = events['block_type'].eq('Blocked Shot').astype(int)
        events['is_blocked_cross'] = events['block_type'].eq('Blocked Cross').astype(int)
        events['is_offside'] = t.isin({'OffsideGiven', 'OffsidePass', 'OffsideProvoked'}).astype(int)
        events['is_caught_offside'] = events['offside_type'].eq('Caught Offside').astype(int)
        events['is_offside_pass'] = events['offside_type'].eq('Offside Pass').astype(int)
        events['is_offside_provoked'] = events['offside_type'].eq('Offside Provoked').astype(int)
        events['is_foul'] = events['foul_type'].notna().astype(int)
        events['is_foul_committed'] = events['foul_type'].eq('Foul Committed').astype(int)
        events['is_foul_suffered'] = events['foul_type'].eq('Foul Suffered').astype(int)
        events['is_aerial_duel'] = t.eq('Aerial').astype(int)
        events['is_aerial_duel_won'] = events['aerial_result'].eq('Won').astype(int)
        events['is_aerial_duel_lost'] = events['aerial_result'].eq('Lost').astype(int)
        events['is_loss_possession'] = (dispossessed_mask | turnover_mask).astype(int)
        events['is_dispossessed'] = events['loss_possession_type'].eq('Dispossessed').astype(int)
        events['is_turnover'] = events['loss_possession_type'].eq('Turnover').astype(int)
        events['is_error'] = error_mask.astype(int)
        events['is_error_leading_to_shot'] = events['error_type'].eq('Error Leading to Shot').astype(int)
        events['is_error_leading_to_goal'] = events['error_type'].eq('Error Leading to Goal').astype(int)
        events['is_goalkeeper'] = events['gk_type'].notna().astype(int)
        events['is_gk_save'] = events['gk_type'].eq('Save').astype(int)
        events['is_gk_claim'] = events['gk_type'].eq('Claim').astype(int)
        events['is_gk_punch'] = events['gk_type'].eq('Punch').astype(int)
        events['is_gk_keeper_pickup'] = events['gk_type'].eq('Keeper Pickup').astype(int)
        events['is_gk_keeper_sweeper'] = events['gk_type'].eq('Keeper Sweeper').astype(int)
        events['is_card'] = events['card_type'].notna().astype(int)
        events['is_yellow_card'] = events['card_type'].eq('Yellow').astype(int)
        events['is_second_yellow_card'] = events['card_type'].eq('Second Yellow').astype(int)
        events['is_red_card'] = events['card_type'].eq('Red').astype(int)
        events['is_substitution'] = t.isin({'SubstitutionOff', 'SubstitutionOn'}).astype(int)
        events['is_substitution_on'] = events['substitution_type'].eq('On').astype(int)
        events['is_substitution_off'] = events['substitution_type'].eq('Off').astype(int)

        events['stat_event_type'] = None
        stat_type_order = [
            (events['is_shot'].eq(1), 'shot'),
            (events['is_pass'].eq(1), 'pass'),
            (events['is_dribble'].eq(1), 'dribble'),
            (events['is_tackle'].eq(1), 'tackle'),
            (events['is_interception'].eq(1), 'interception'),
            (events['is_clearance'].eq(1), 'clearance'),
            (events['is_block'].eq(1), 'block'),
            (events['is_offside'].eq(1), 'offside'),
            (events['is_foul'].eq(1), 'foul'),
            (events['is_aerial_duel'].eq(1), 'aerial_duel'),
            (events['is_loss_possession'].eq(1), 'loss_possession'),
            (events['is_error'].eq(1), 'error'),
            (events['is_goalkeeper'].eq(1), 'goalkeeper'),
            (events['is_card'].eq(1), 'card'),
            (events['is_substitution'].eq(1), 'substitution'),
            (events['is_ball_recovery'].eq(1), 'ball_recovery'),
        ]
        for mask, label in stat_type_order:
            events.loc[mask, 'stat_event_type'] = label

        raw_touch = pd.to_numeric(events.get('is_touch', 0), errors='coerce').fillna(0).astype(int)
        events['is_touch'] = raw_touch
        relevant_mask = events['stat_event_type'].notna() | events['is_touch'].eq(1) | events['is_key_pass'].eq(1)
        processed_events = events.loc[relevant_mask].reset_index(drop=True).copy()

        drop_cols = ['_id', 'qualifiers', 'qualifier_names', 'qset', 'qmap', 'qsp', 'qst', 'qmpa']
        processed_events = processed_events.drop(columns=[c for c in drop_cols if c in processed_events.columns])

        def as_bool(source_col: str) -> pd.Series:
            if source_col not in processed_events.columns:
                return pd.Series(False, index=processed_events.index)
            return pd.to_numeric(processed_events[source_col], errors='coerce').fillna(0).astype(int).astype(bool)

        final_boolean_sources = {
            'shot_event': 'is_shot',
            'pass_event': 'is_pass',
            'pass_attempt': 'is_pass_attempt',
            'pass_completed': 'pass_successful',
            'pass_incomplete': 'is_pass_incomplete',
            'dribble_event': 'is_dribble',
            'tackle_attempted_event': 'is_tackle',
            'interception_event': 'is_interception',
            'clearance_event': 'is_clearance',
            'block_event': 'is_block',
            'offside_event': 'is_offside',
            'foul_event': 'is_foul',
            'aerial_duel_event': 'is_aerial_duel',
            'touch_event': 'is_touch',
            'loss_possession_event': 'is_loss_possession',
            'error_event': 'is_error',
            'goalkeeper_event': 'is_goalkeeper',
            'save_event': 'is_gk_save',
            'claim_event': 'is_gk_claim',
            'punch_event': 'is_gk_punch',
            'ball_recovery_event': 'is_ball_recovery',
            'card_event': 'is_card',
            'substitution_event': 'is_substitution',
            'shot_goal': 'shot_is_goal',
            'shot_on_target': 'shot_on_target',
            'shot_off_target': 'shot_off_target',
            'shot_woodwork': 'shot_woodwork',
            'shot_blocked': 'shot_blocked',
            'shot_own_goal': 'shot_own_goal',
            'goal_assist': 'goal_assist',
            'pass_long': 'is_pass_long',
            'pass_short': 'is_pass_short',
            'pass_chipped': 'is_pass_chipped',
            'pass_ground': 'is_pass_ground',
            'pass_head': 'is_pass_head',
            'pass_feet': 'is_pass_feet',
            'pass_forward': 'pass_forward',
            'pass_backward': 'pass_backward',
            'pass_left': 'pass_left',
            'pass_right': 'pass_right',
            'pass_defensive_third': 'is_pass_defensive_third',
            'pass_mid_third': 'is_pass_mid_third',
            'pass_final_third': 'is_pass_final_third',
            'pass_cross': 'pass_cross',
            'pass_freekick': 'pass_freekick',
            'pass_corner': 'pass_corner',
            'pass_through_ball': 'pass_through_ball',
            'pass_throw_in': 'pass_throw_in',
            'pass_key_pass_qualifier': 'pass_key_pass_qualifier',
            'pass_key_pass': 'is_key_pass',
            'dribble_successful': 'is_dribble_successful',
            'dribble_unsuccessful': 'is_dribble_unsuccessful',
            'dispossessed': 'is_dispossessed',
            'turnover': 'is_turnover',
            'tackle_gained_possession': 'is_tackle_gained_possession',
            'tackle_did_not_get_possession': 'is_tackle_did_not_get_possession',
            'tackle_was_dribbled': 'is_tackle_was_dribbled',
            'clearance_head': 'is_clearance_head',
            'clearance_feet': 'is_clearance_feet',
            'blocked_shot': 'is_blocked_shot',
            'blocked_cross': 'is_blocked_cross',
            'caught_offside': 'is_caught_offside',
            'offside_pass': 'is_offside_pass',
            'offside_provoked': 'is_offside_provoked',
            'foul_committed': 'is_foul_committed',
            'foul_suffered': 'is_foul_suffered',
            'aerial_duel_won': 'is_aerial_duel_won',
            'aerial_duel_lost': 'is_aerial_duel_lost',
            'error_leading_to_shot': 'is_error_leading_to_shot',
            'error_leading_to_goal': 'is_error_leading_to_goal',
            'keeper_pickup': 'is_gk_keeper_pickup',
            'keeper_sweeper': 'is_gk_keeper_sweeper',
            'yellow_card': 'is_yellow_card',
            'second_yellow_card': 'is_second_yellow_card',
            'red_card': 'is_red_card',
            'substitution_on': 'is_substitution_on',
            'substitution_off': 'is_substitution_off',
        }
        boolean_data = {final_col: as_bool(source_col) for final_col, source_col in final_boolean_sources.items()}
        boolean_data.update({
            'shot_zone_6_yard_box': processed_events['shot_zone'].eq('6-yard box'),
            'shot_zone_penalty_area': processed_events['shot_zone'].eq('Penalty Area'),
            'shot_zone_outside_box': processed_events['shot_zone'].eq('Outside of box'),
            'shot_open_play': processed_events['shot_situation'].eq('Open Play'),
            'shot_fastbreak': processed_events['shot_situation'].eq('Fastbreak'),
            'shot_set_piece': processed_events['shot_situation'].eq('Set Pieces'),
            'shot_penalty': processed_events['shot_situation'].eq('Penalty'),
            'shot_right_foot': processed_events['shot_body_part'].eq('Right foot'),
            'shot_left_foot': processed_events['shot_body_part'].eq('Left foot'),
            'shot_head': processed_events['shot_body_part'].eq('Head'),
            'shot_other_body_part': processed_events['shot_body_part'].eq('Other body parts'),
        })
        boolean_frame = pd.DataFrame(boolean_data, index=processed_events.index)
        processed_events = processed_events.drop(columns=[c for c in boolean_frame.columns if c in processed_events.columns])
        processed_events = pd.concat([processed_events, boolean_frame], axis=1)

        old_helper_flags = [c for c in processed_events.columns if c.startswith('is_')]
        old_helper_flags.extend(['shot_is_goal', 'shot_assisted_key_pass', 'pass_successful'])
        processed_events = processed_events.drop(columns=[c for c in old_helper_flags if c in processed_events.columns])

        ordered_cols = [
            'game_id', 'season', 'competition_country', 'competition_name', 'game_date', 'game_status', 'week',
            'home_team_id', 'home_team_name', 'away_team_id', 'away_team_name',
            'event_idx', 'period', 'minute', 'second', 'expanded_minute',
            'team_id', 'team', 'player_id', 'player', 'type', 'outcome_type',
            'x', 'y', 'end_x', 'end_y', 'goal_mouth_y', 'goal_mouth_z', 'blocked_x', 'blocked_y',
            'related_event_id', 'related_player_id', 'stat_event_type',
            'shot_event', 'pass_event', 'pass_completed', 'dribble_event', 'tackle_attempted_event',
            'interception_event', 'clearance_event', 'block_event',
            'offside_event', 'foul_event', 'aerial_duel_event', 'touch_event', 'loss_possession_event',
            'error_event', 'save_event', 'claim_event', 'punch_event',
            'goalkeeper_event', 'ball_recovery_event', 'card_event', 'substitution_event',
            'shot_goal', 'shot_on_target', 'shot_off_target', 'shot_woodwork', 'shot_blocked',
            'shot_own_goal', 'goal_assist',
            'shot_zone_6_yard_box', 'shot_zone_penalty_area', 'shot_zone_outside_box',
            'shot_open_play', 'shot_fastbreak', 'shot_set_piece', 'shot_penalty',
            'shot_right_foot', 'shot_left_foot', 'shot_head', 'shot_other_body_part',
            'shot_result', 'shot_zone', 'shot_situation', 'shot_body_part',
            'pass_attempt', 'pass_incomplete',
            'pass_cross', 'pass_freekick', 'pass_corner', 'pass_through_ball', 'pass_throw_in',
            'pass_key_pass', 'pass_key_pass_qualifier',
            'pass_long', 'pass_short', 'pass_length',
            'pass_chipped', 'pass_ground', 'pass_height',
            'pass_head', 'pass_feet', 'pass_body_part',
            'pass_forward', 'pass_backward', 'pass_left', 'pass_right',
            'pass_defensive_third', 'pass_mid_third', 'pass_final_third', 'pass_target_zone',
            'dribble_successful', 'dribble_unsuccessful',
            'tackle_gained_possession', 'tackle_did_not_get_possession', 'tackle_was_dribbled', 'tackle_result',
            'clearance_head', 'clearance_feet', 'clearance_body_part',
            'blocked_shot', 'blocked_cross', 'block_type',
            'caught_offside', 'offside_pass', 'offside_provoked', 'offside_type',
            'foul_committed', 'foul_suffered', 'foul_type',
            'aerial_duel_won', 'aerial_duel_lost', 'aerial_result',
            'dispossessed', 'turnover', 'loss_possession_type',
            'error_leading_to_shot', 'error_leading_to_goal', 'error_type',
            'keeper_pickup', 'keeper_sweeper', 'gk_type',
            'yellow_card', 'second_yellow_card', 'red_card', 'card_type',
            'substitution_on', 'substitution_off', 'substitution_type',
        ]
        ordered_cols = [c for c in ordered_cols if c in processed_events.columns]
        remaining_cols = [c for c in processed_events.columns if c not in ordered_cols]
        return processed_events[ordered_cols + remaining_cols]

    def save_processed_raw_games(self, limit: Optional[int] = None) -> Dict[str, int]:
        raw_ids = self._raw_game_ids()
        processed_ids = self._processed_game_ids()
        pending = sorted(raw_ids - processed_ids)
        if limit is not None:
            pending = pending[:limit]

        if not pending:
            logger.info('No raw games pending processed event generation.')
            self.client.close()
            return {'processed_games': 0, 'inserted_rows': 0}

        inserted_games = 0
        inserted_rows_total = 0
        for game_id in pending:
            try:
                logger.info('Processing game events for game_id=%s', game_id)
                raw_docs = list(
                    self.collection_raw_events.find(
                        {'season': self.config.season.year, 'game_id': game_id},
                        {'_id': 0},
                    )
                )
                raw_events = pd.DataFrame(raw_docs)
                if raw_events.empty:
                    logger.info('No raw event rows found for game_id=%s', game_id)
                    continue

                processed_events = self.build_processed_game_events(raw_events)
                records = self._records_for_mongo(processed_events)
                if not records:
                    logger.info('No processed event rows produced for game_id=%s', game_id)
                    continue

                self.collection_processed_events.delete_many({'season': self.config.season.year, 'game_id': game_id})
                try:
                    result = self.collection_processed_events.insert_many(records, ordered=False)
                    inserted_count = len(result.inserted_ids)
                except BulkWriteError as bwe:
                    inserted_count = bwe.details.get('nInserted', 0)
                    self.collection_processed_events.delete_many({'season': self.config.season.year, 'game_id': game_id})
                    logger.warning(
                        'BulkWriteError for processed game_id=%s, inserted %s rows before failure',
                        game_id,
                        inserted_count,
                    )
                    continue

                inserted_games += 1
                inserted_rows_total += inserted_count
                logger.info('Inserted %s processed rows for game_id=%s', inserted_count, game_id)
            except Exception:
                logger.exception('Error processing game events for game_id=%s', game_id)

        self.client.close()
        return {'processed_games': inserted_games, 'inserted_rows': inserted_rows_total}


def main(config_path: Path | str = CONFIG_PATH) -> Dict[str, int]:
    setup_logging()
    config = load_app_config(str(config_path))
    events = ProcessedEvents(config)
    return events.save_processed_raw_games()


if __name__ == '__main__':
    main()
