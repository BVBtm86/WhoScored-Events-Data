from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Set

import numpy as np
import pandas as pd
from pymongo import MongoClient
from pymongo.errors import BulkWriteError

from helper import setup_logging, load_app_config, logger
from config import ScrapeDataConfig

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'


class GameStats:

    def __init__(self, config: ScrapeDataConfig):
        self.config = config
        self.client = MongoClient(config.mongo.url)
        self.db = self.client[config.mongo.db]
        self.collection_processed_events = self.db[config.mongo.collection['collection_processed_events']]
        self.collection_team_stats = self.db[config.mongo.collection['collection_team_game_stats']]
        self.collection_player_stats = self.db[config.mongo.collection['collection_player_game_stats']]

        self.team_match_features = config.game_stats.team_match_features
        self.player_match_features = config.game_stats.player_match_features
        self.stat_columns = config.game_stats.match_stats
        self.pass_completion_base_columns = config.game_stats.pass_complettion_stats
        self.perc_stats = config.game_stats.perc_stats
        self.STAT_MATCH_RENAME = config.game_stats.match_stats_rename

    def _processed_game_ids(self) -> Set[int]:
        return {
            int(x)
            for x in self.collection_processed_events.distinct('game_id', {'season': self.config.season.year})
            if x is not None
        }

    def _stats_game_ids(self) -> Set[int]:
        team_ids = {
            int(x)
            for x in self.collection_team_stats.distinct('game_id', {'season': self.config.season.year})
            if x is not None
        }
        player_ids = {
            int(x)
            for x in self.collection_player_stats.distinct('game_id', {'season': self.config.season.year})
            if x is not None
        }
        return team_ids & player_ids

    @staticmethod
    def _normalize_mongo_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return {k: GameStats._normalize_mongo_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [GameStats._normalize_mongo_value(v) for v in value]
        if isinstance(value, tuple):
            return [GameStats._normalize_mongo_value(v) for v in value]
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

    @staticmethod
    def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        denominator = denominator.replace(0, np.nan)
        return (numerator / denominator).replace([np.inf, -np.inf], np.nan)

    def _add_rate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if {'shot_zone_6_yard_box', 'shot_zone_penalty_area'}.issubset(df.columns):
            df['shot_zone_inside_box'] = df['shot_zone_6_yard_box'] + df['shot_zone_penalty_area']
        if {'shot_right_foot', 'shot_left_foot'}.issubset(df.columns):
            df['shot_foot'] = df['shot_right_foot'] + df['shot_left_foot']

        for new_col, formula in self.perc_stats.items():
            numerator = formula['numerator']
            denominator = formula['denominator']
            if numerator not in df.columns or denominator not in df.columns:
                df[new_col] = np.nan
                continue
            df[new_col] = self._safe_div(df[numerator], df[denominator])
        return df

    def _prepare_game_events(self, events: pd.DataFrame) -> pd.DataFrame:
        if events.empty:
            return events

        events = events.copy()
        home_team_id = events['home_team_id'].iloc[0]
        away_team_id = events['away_team_id'].iloc[0]
        home_team_name = events['home_team_name'].iloc[0]
        away_team_name = events['away_team_name'].iloc[0]

        events['game_venue'] = events['team_id'].map({
            home_team_id: 'home',
            away_team_id: 'away',
        }).fillna('neutral')
        events['team_name'] = events['team_id'].map({
            home_team_id: home_team_name,
            away_team_id: away_team_name,
        }).fillna(events['team'])
        events['opponent_team_id'] = events['team_id'].map({
            home_team_id: away_team_id,
            away_team_id: home_team_id,
        })
        events['opponent_team_name'] = events['team_id'].map({
            home_team_id: away_team_name,
            away_team_id: home_team_name,
        })

        foul_mask = events['type'].eq('Foul')
        foul_committed = foul_mask & events['outcome_type'].eq('Unsuccessful')
        foul_suffered = foul_mask & events['outcome_type'].eq('Successful')
        events.loc[foul_mask, 'foul_committed'] = foul_committed[foul_mask]
        events.loc[foul_mask, 'foul_suffered'] = foul_suffered[foul_mask]
        if 'foul_type' in events.columns:
            events.loc[foul_committed, 'foul_type'] = 'Foul Committed'
            events.loc[foul_suffered, 'foul_type'] = 'Foul Suffered'

        return events

    def _add_derived_event_stats(self, game_events: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        game_events = game_events.copy()
        derived_columns: list[str] = []
        if 'pass_completed' not in game_events.columns:
            return game_events, derived_columns

        pass_completed = pd.to_numeric(game_events['pass_completed'], errors='coerce').fillna(0).astype(int)
        for col in self.pass_completion_base_columns:
            if col not in game_events.columns:
                continue
            completed_col = f'{col}_completed'
            base = pd.to_numeric(game_events[col], errors='coerce').fillna(0).astype(int)
            game_events[completed_col] = (base.eq(1) & pass_completed.eq(1)).astype(int)
            derived_columns.append(completed_col)
        return game_events, derived_columns

    def _rename_output_stat_keys(self, stats: dict[str, Any]) -> dict[str, Any]:
        return {self.STAT_MATCH_RENAME.get(key, key): value for key, value in stats.items()}

    def _rename_interval_stat_keys(
        self,
        interval_lookup: dict[tuple[int, int], dict[str, dict[str, Any]]],
    ) -> dict[tuple[int, int], dict[str, dict[str, Any]]]:
        for interval_stats in interval_lookup.values():
            for interval, stats in interval_stats.items():
                interval_stats[interval] = self._rename_output_stat_keys(stats)
        return interval_lookup

    @staticmethod
    def _interval_masks(game_events: pd.DataFrame) -> dict[str, pd.Series]:
        minute = pd.to_numeric(game_events['minute'], errors='coerce')
        period = game_events['period'].fillna('')
        first_half = period.eq('FirstHalf')
        second_half = period.eq('SecondHalf')
        return {
            'ft': pd.Series(True, index=game_events.index),
            'fh': first_half,
            'sh': second_half,
            'm_1_15': minute.lt(15),
            'm_16_30': minute.ge(15) & minute.lt(30),
            'm_31_45': first_half & minute.ge(30),
            'm_46_60': second_half & minute.lt(60),
            'm_61_75': second_half & minute.ge(60) & minute.lt(75),
            'm_76_90': second_half & minute.ge(75),
            'm_1_30': minute.lt(30),
            'm_16_45': first_half & minute.ge(15),
            'm_31_60': (first_half & minute.ge(30)) | (second_half & minute.lt(60)),
            'm_46_75': second_half & minute.lt(75),
            'm_61_90': second_half & minute.ge(60),
        }

    def _add_possession_stats(self, team_stats: pd.DataFrame) -> pd.DataFrame:
        team_stats = team_stats.copy()
        if 'pass_attempt' not in team_stats.columns:
            team_stats['possession'] = np.nan
            return team_stats

        pass_total = team_stats.groupby('game_id')['pass_attempt'].transform('sum')
        team_stats['possession'] = self._safe_div(team_stats['pass_attempt'], pass_total)
        return team_stats

    @staticmethod
    def _stat_dict(row: pd.Series, stat_value_columns: list[str]) -> dict[str, Any]:
        out = {}
        for col in stat_value_columns:
            value = row.get(col)
            if isinstance(value, np.generic):
                value = value.item()
            try:
                if pd.isna(value):
                    value = None
            except Exception:
                pass
            out[col] = value
        return out

    def _summarize_by_team(self, game_events: pd.DataFrame, stat_columns: list[str]) -> pd.DataFrame:
        game_events, derived_columns = self._add_derived_event_stats(game_events)
        effective_stat_columns = [c for c in [*stat_columns, *derived_columns] if c in game_events.columns]
        game_events[effective_stat_columns] = (
            game_events[effective_stat_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
        )
        team_stats = (
            game_events.groupby(self.team_match_features, dropna=False)[effective_stat_columns]
            .sum()
            .reset_index()
        )
        team_stats = self._align_team_fouls(team_stats)
        team_stats = self._add_rate_columns(team_stats)
        return self._add_possession_stats(team_stats)

    @staticmethod
    def _align_team_fouls(team_stats: pd.DataFrame) -> pd.DataFrame:
        if not {'game_id', 'team_id', 'opponent_team_id', 'foul_committed', 'foul_suffered'}.issubset(team_stats.columns):
            return team_stats

        team_stats = team_stats.copy()
        committed_lookup = team_stats.set_index(['game_id', 'team_id'])['foul_committed']
        opponent_keys = pd.MultiIndex.from_frame(team_stats[['game_id', 'opponent_team_id']])
        team_stats['foul_suffered'] = committed_lookup.reindex(opponent_keys).fillna(0).to_numpy()
        return team_stats

    def _add_score_stats(
        self,
        interval_lookup: dict[tuple[int, int], dict[str, dict[str, Any]]],
        game_events: pd.DataFrame,
    ) -> None:
        team_pairs = game_events[['game_id', 'team_id', 'opponent_team_id']].drop_duplicates()
        for _, team_row in team_pairs.iterrows():
            key = (int(team_row['game_id']), int(team_row['team_id']))
            opponent_key = (int(team_row['game_id']), int(team_row['opponent_team_id']))
            for interval, stats in interval_lookup.get(key, {}).items():
                opponent_stats = interval_lookup.get(opponent_key, {}).get(interval, {})
                goals = int(stats.get('shot_goal', 0) or 0) + int(opponent_stats.get('shot_own_goal', 0) or 0)
                possession = stats.pop('possession', None)
                stats.pop('goals_for', None)
                reordered_stats = {}
                if possession is not None:
                    reordered_stats['possession'] = possession
                reordered_stats['goals'] = goals
                reordered_stats.update(stats)
                stats.clear()
                stats.update(reordered_stats)

    def _build_team_interval_stats(
        self,
        game_events: pd.DataFrame,
        stat_columns: list[str],
    ) -> dict[tuple[int, int], dict[str, dict[str, Any]]]:
        interval_lookup: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
        for interval, mask in self._interval_masks(game_events).items():
            interval_events = game_events.loc[mask].copy()
            if interval_events.empty:
                continue
            interval_stats = self._summarize_by_team(interval_events, stat_columns)
            stat_value_columns = [c for c in interval_stats.columns if c not in self.team_match_features]
            for _, row in interval_stats.iterrows():
                key = (int(row['game_id']), int(row['team_id']))
                interval_lookup.setdefault(key, {})[interval] = self._stat_dict(row, stat_value_columns)
        self._add_score_stats(interval_lookup, game_events)
        return self._rename_interval_stat_keys(interval_lookup)

    def build_game_team_stats(self, game_events: pd.DataFrame, stat_columns: list[str]) -> pd.DataFrame:
        feature_columns = [c for c in self.team_match_features if c in game_events.columns]
        if not feature_columns:
            raise ValueError('No valid team match feature columns found in config game_stats.team_match_features.')

        team_stats = game_events[feature_columns].drop_duplicates().reset_index(drop=True).copy()
        interval_lookup = self._build_team_interval_stats(game_events, stat_columns)
        team_stats['stats'] = team_stats.apply(
            lambda row: interval_lookup.get((int(row['game_id']), int(row['team_id'])), {}),
            axis=1,
        )
        team_stats['opp_stats'] = team_stats.apply(
            lambda row: interval_lookup.get((int(row['game_id']), int(row['opponent_team_id'])), {}),
            axis=1,
        )
        return team_stats[feature_columns + ['stats', 'opp_stats']]

    @staticmethod
    def _event_match_seconds(game_events: pd.DataFrame) -> pd.Series:
        minute = pd.to_numeric(game_events['minute'], errors='coerce').fillna(0)
        second = pd.to_numeric(game_events.get('second', 0), errors='coerce').fillna(0)
        return minute.mul(60).add(second)

    def _add_player_appearance_columns(self, game_events: pd.DataFrame) -> pd.DataFrame:
        game_events = game_events.copy()
        if game_events.empty or not {'game_id', 'team_id', 'player_id'}.issubset(game_events.columns):
            return game_events

        event_seconds = self._event_match_seconds(game_events)
        match_end_seconds = event_seconds.groupby(game_events['game_id']).transform('max')
        player_key = ['game_id', 'team_id', 'player_id']

        sub_on_seconds = event_seconds.where(game_events['type'].eq('SubstitutionOn')).groupby(
            [game_events[c] for c in player_key], dropna=False
        ).min()
        sub_off_seconds = event_seconds.where(game_events['type'].eq('SubstitutionOff')).groupby(
            [game_events[c] for c in player_key], dropna=False
        ).min()

        appearance = game_events[player_key].dropna().drop_duplicates().copy()
        appearance = appearance.merge(sub_on_seconds.rename('sub_on_seconds').reset_index(), on=player_key, how='left')
        appearance = appearance.merge(
            sub_off_seconds.rename('sub_off_seconds').reset_index(),
            on=player_key,
            how='left',
        )
        appearance['starting_lineup'] = appearance['sub_on_seconds'].isna()

        match_end_by_game = game_events.assign(_match_end_seconds=match_end_seconds)[
            ['game_id', '_match_end_seconds']
        ].drop_duplicates()
        appearance = appearance.merge(match_end_by_game, on='game_id', how='left')
        appearance['start_seconds'] = appearance['sub_on_seconds'].fillna(0)
        appearance['end_seconds'] = appearance['sub_off_seconds'].fillna(appearance['_match_end_seconds'])
        appearance['minutes_played'] = (
            (appearance['end_seconds'] - appearance['start_seconds']).clip(lower=0).div(60).apply(np.ceil).astype(int)
        )

        appearance = appearance[player_key + ['starting_lineup', 'minutes_played']]
        game_events = game_events.merge(appearance, on=player_key, how='left')
        game_events['starting_lineup'] = game_events['starting_lineup'].fillna(False).astype(bool)
        game_events['minutes_played'] = pd.to_numeric(game_events['minutes_played'], errors='coerce').astype('Int64')
        return game_events

    @staticmethod
    def _is_per_90_source_stat(stat_name: str, value: Any) -> bool:
        return True

    def _add_player_per_90_stats(self, stats: dict[str, dict[str, Any]], minutes_played: Any) -> dict[str, dict[str, Any]]:
        if 'ft' not in stats:
            return stats

        minutes = pd.to_numeric(pd.Series([minutes_played]), errors='coerce').iloc[0]
        minutes = None if pd.isna(minutes) else float(minutes)
        per_90_stats: dict[str, Any] = {}
        for source_col, raw_value in stats['ft'].items():
            if not self._is_per_90_source_stat(source_col, raw_value):
                continue
            value = pd.to_numeric(pd.Series([raw_value]), errors='coerce').iloc[0]
            if pd.isna(value) or minutes is None or minutes == 0:
                per_90_stats[source_col] = None
            else:
                per_90_stats[source_col] = float(value) / minutes * 90

        stats['per_90'] = per_90_stats
        return stats

    def _summarize_by_player(self, game_events: pd.DataFrame, stat_columns: list[str]) -> pd.DataFrame:
        player_events = game_events[game_events['player_id'].notna()].copy()
        if player_events.empty:
            return pd.DataFrame()

        player_events, derived_columns = self._add_derived_event_stats(player_events)
        effective_stat_columns = [c for c in [*stat_columns, *derived_columns] if c in player_events.columns]
        feature_columns = [c for c in self.player_match_features if c in player_events.columns]
        if not feature_columns:
            raise ValueError('No valid player match feature columns found in config game_stats.player_match_features.')
        if not effective_stat_columns:
            return player_events[feature_columns].drop_duplicates().reset_index(drop=True)

        player_events[effective_stat_columns] = (
            player_events[effective_stat_columns].apply(pd.to_numeric, errors='coerce').fillna(0)
        )
        player_stats = (
            player_events.groupby(feature_columns, dropna=False)[effective_stat_columns]
            .sum()
            .reset_index()
        )
        return self._add_rate_columns(player_stats)

    def _build_player_interval_stats(
        self,
        game_events: pd.DataFrame,
        stat_columns: list[str],
    ) -> dict[tuple[int, int], dict[str, dict[str, Any]]]:
        interval_lookup: dict[tuple[int, int], dict[str, dict[str, Any]]] = {}
        for interval, mask in self._interval_masks(game_events).items():
            interval_events = game_events.loc[mask].copy()
            if interval_events.empty:
                continue

            interval_stats = self._summarize_by_player(interval_events, stat_columns)
            if interval_stats.empty:
                continue

            feature_columns = [c for c in self.player_match_features if c in interval_stats.columns]
            stat_value_columns = [c for c in interval_stats.columns if c not in feature_columns]
            for _, row in interval_stats.iterrows():
                key = (int(row['game_id']), int(row['player_id']))
                interval_lookup.setdefault(key, {})[interval] = self._stat_dict(row, stat_value_columns)
        return interval_lookup

    def build_game_player_stats(self, game_events: pd.DataFrame, stat_columns: list[str]) -> pd.DataFrame:
        player_events = self._add_player_appearance_columns(game_events)
        player_events = player_events[player_events['player_id'].notna()].copy()
        feature_columns = [c for c in self.player_match_features if c in player_events.columns]
        if player_events.empty:
            return pd.DataFrame(columns=feature_columns + ['stats'])
        if not feature_columns:
            raise ValueError('No valid player match feature columns found in config game_stats.player_match_features.')

        players = player_events[feature_columns].drop_duplicates().reset_index(drop=True).copy()
        interval_lookup = self._build_player_interval_stats(player_events, stat_columns)
        players['stats'] = players.apply(
            lambda row: interval_lookup.get((int(row['game_id']), int(row['player_id'])), {}),
            axis=1,
        )
        players['stats'] = players.apply(
            lambda row: self._add_player_per_90_stats(row['stats'], row.get('minutes_played')),
            axis=1,
        )
        players['stats'] = players['stats'].apply(
            lambda stats: self._rename_interval_stat_keys({(0, 0): stats})[(0, 0)]
        )
        return players[feature_columns + ['stats']]

    def _load_processed_game_events(self, game_id: int) -> pd.DataFrame:
        docs = list(
            self.collection_processed_events.find(
                {'season': self.config.season.year, 'game_id': int(game_id)},
                {'_id': 0},
            )
        )
        return pd.DataFrame(docs)

    def save_game_stats(self, limit: Optional[int] = None) -> Dict[str, int]:
        processed_ids = self._processed_game_ids()
        stats_ids = self._stats_game_ids()
        pending = sorted(processed_ids - stats_ids)
        if limit is not None:
            pending = pending[:limit]

        if not pending:
            logger.info('No processed games pending game stat generation.')
            self.client.close()
            return {'processed_games': 0, 'team_rows': 0, 'player_rows': 0}

        processed_games = 0
        team_rows_total = 0
        player_rows_total = 0
        for game_id in pending:
            try:
                logger.info('Building game stats for game_id=%s', game_id)
                game_events = self._load_processed_game_events(game_id)
                if game_events.empty:
                    logger.info('No processed event rows found for game_id=%s', game_id)
                    continue

                game_events = self._prepare_game_events(game_events)
                stat_columns = [c for c in self.stat_columns if c in game_events.columns]
                team_stats = self.build_game_team_stats(game_events, stat_columns)
                player_stats = self.build_game_player_stats(game_events, stat_columns)

                team_records = self._records_for_mongo(team_stats)
                player_records = self._records_for_mongo(player_stats)
                self.collection_team_stats.delete_many({'season': self.config.season.year, 'game_id': game_id})
                self.collection_player_stats.delete_many({'season': self.config.season.year, 'game_id': game_id})

                team_inserted = 0
                player_inserted = 0
                try:
                    if team_records:
                        result = self.collection_team_stats.insert_many(team_records, ordered=False)
                        team_inserted = len(result.inserted_ids)
                    if player_records:
                        result = self.collection_player_stats.insert_many(player_records, ordered=False)
                        player_inserted = len(result.inserted_ids)
                except BulkWriteError as bwe:
                    self.collection_team_stats.delete_many({'season': self.config.season.year, 'game_id': game_id})
                    self.collection_player_stats.delete_many({'season': self.config.season.year, 'game_id': game_id})
                    logger.warning(
                        'BulkWriteError for game stats game_id=%s, details=%s',
                        game_id,
                        bwe.details,
                    )
                    continue

                processed_games += 1
                team_rows_total += team_inserted
                player_rows_total += player_inserted
                logger.info(
                    'Inserted game stats for game_id=%s: team_rows=%s player_rows=%s',
                    game_id,
                    team_inserted,
                    player_inserted,
                )
            except Exception:
                logger.exception('Error building game stats for game_id=%s', game_id)

        self.client.close()
        return {
            'processed_games': processed_games,
            'team_rows': team_rows_total,
            'player_rows': player_rows_total,
        }


def main(config_path: Path | str = CONFIG_PATH) -> Dict[str, int]:
    setup_logging()
    config = load_app_config(str(config_path))
    stats = GameStats(config)
    return stats.save_game_stats()


if __name__ == '__main__':
    main()
