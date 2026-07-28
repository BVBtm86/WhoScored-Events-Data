from __future__ import annotations

import types
from pathlib import Path
from typing import Any, Dict, Optional, Set

import pandas as pd
import soccerdata as sd
from pymongo import MongoClient, UpdateOne

from prod_pipeline.helper import setup_logging, patch_soccerdata, load_app_config, logger
from prod_pipeline.config import ScrapeDataConfig

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'


def infer_formation(position_string: str) -> str:
    positions = [position for position in position_string.split("-") if position and position != "GK"]

    defenders = sum(position in {"DR", "DC", "DL"} for position in positions)
    defensive_midfielders = sum(position in {"DMR", "DMC", "DML"} for position in positions)
    midfielders = sum(position in {"MR", "MC", "ML"} for position in positions)
    attacking_midfielders = sum(position in {"AMR", "AMC", "AML"} for position in positions)
    forwards = sum(position in {"FWR", "FW", "FWL"} for position in positions)

    if attacking_midfielders and forwards:
        middle = defensive_midfielders + midfielders
        return "-".join(
            str(number)
            for number in [defenders, middle, attacking_midfielders, forwards]
            if number > 0
        )

    midfield = defensive_midfielders + midfielders
    return "-".join(
        str(number)
        for number in [defenders, midfield, forwards]
        if number > 0
    )


class GameLineups:
    def __init__(self, config: ScrapeDataConfig):
        self.config = config
        patch_soccerdata()
        self.ws = sd.WhoScored(
            leagues=config.season.league,
            seasons=config.season.year,
        )
        self.client = MongoClient(config.mongo.url)
        self.db = self.client[config.mongo.db]
        self.collection_schedule = self.db[config.mongo.collections['collection_schedule']]
        self.collection_lineups = self.db[config.mongo.collections['collection_lineups']]
        self.collection_formations = self.db[config.mongo.collections['collection_formations']]
        self.collection_teams = self.db[config.mongo.collections['collection_teams']]
        self.formation_mapping = config.formation_mapping

    def _finished_games_df(self) -> pd.DataFrame:
        docs = list(
            self.collection_schedule.find(
                {'season': self.config.season.year, 'game_status': 'finished'},
                {'_id': 0},
            ).sort('game_date', 1)
        )
        return pd.DataFrame(docs)

    def _available_teams_df(self) -> pd.DataFrame:

        available_teams = pd.DataFrame(
            self.collection_teams.find(
                {},
                {'_id': 0, 'ws_team_id': 1, 'ws_team_name': 1})
        )

        available_teams.rename(columns={'ws_team_id': 'team_id', 'ws_team_name': 'team_name'}, inplace=True)
        return available_teams
    
    def _lineups_extracted(self) -> Set[int]:
        lineup_ids = {
            int(x)
            for x in self.collection_lineups.distinct(
                'game_id',
                {'season': self.config.season.year},
            )
            if x is not None
        }
        formation_ids = {
            int(x)
            for x in self.collection_formations.distinct(
                'game_id',
                {'season': self.config.season.year},
            )
            if x is not None
        }
        return lineup_ids & formation_ids if formation_ids else set()

    def _patch_ws_schedule(self, finished_df: pd.DataFrame) -> None:
        def mongo_finished_schedule(ws_self, force_cache=False):
            return pd.DataFrame(
                {
                    'league': self.config.season.league,
                    'season': self.config.season.year_short,
                    'game': finished_df['game_id'].map(lambda x: f"match_{int(x)}"),
                    'game_id': finished_df['game_id'].astype(int),
                }
            )

        self.ws.read_schedule = types.MethodType(mongo_finished_schedule, self.ws)

    @staticmethod
    def _normalize_mongo_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return {k: GameLineups._normalize_mongo_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [GameLineups._normalize_mongo_value(v) for v in value]
        if isinstance(value, tuple):
            return [GameLineups._normalize_mongo_value(v) for v in value]
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

    def _process_game_info(self, df_players: pd.DataFrame, game_info: pd.DataFrame) -> pd.DataFrame:
        df = df_players.copy()
        df['_source_order'] = range(len(df))
        df = df.rename(
            columns={
                'is_starter': 'starting_lineup',
            }
        )

        game_id = int(game_info['game_id'].iloc[0])
        df['game_id'] = game_id
        df['season'] = self.config.season.year
        df['competition_name'] = self.config.season.name
        df['competition_country'] = self.config.season.country

        for col in ['game_id', 'team_id', 'player_id', 'minutes_played', 'jersey_number']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

        if 'starting_lineup' in df.columns:
            df['starting_lineup'] = df['starting_lineup'].fillna(False).astype(bool)

        metadata_cols = [
            'game_id',
            'game_date',
            'week',
            'home_team_id',

        ]
        metadata = game_info[[c for c in metadata_cols if c in game_info.columns]].drop_duplicates('game_id')
        df = df.merge(metadata, on='game_id', how='left')

        if 'minutes_played' in df.columns:
            df['played'] = pd.to_numeric(df['minutes_played'], errors='coerce').fillna(0).gt(0)
        else:
            df['played'] = False

        if {'team_id', 'home_team_id'}.issubset(df.columns):
            df['game_venue'] = df['team_id'].eq(df['home_team_id']).map({True: 'home', False: 'away'})
        else:
            df['game_venue'] = None

        df = pd.merge(
            left=df,
            right=self._available_teams_df(),
            on='team_id',
            how='left'
        )
        return df[self.config.game_lineups]

    def _known_formation_patterns(self) -> Set[str]:
        return {
            str(x)
            for x in self.collection_formations.distinct('positions')
            if x
        }

    def _build_formations(
        self,
        lineups: pd.DataFrame,
        known_patterns: Set[str],
    ) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
        if lineups.empty:
            return pd.DataFrame(columns=self.config.game_formations), []

        starters = lineups.loc[lineups['starting_lineup'].fillna(False)].copy()
        if starters.empty:
            return pd.DataFrame(columns=self.config.game_formations), []

        if '_source_order' not in starters.columns:
            starters['_source_order'] = range(len(starters))

        formation_rows: list[dict[str, Any]] = []
        unknown_patterns: list[dict[str, Any]] = []

        for _, team_data in starters.sort_values('_source_order').groupby('team_id', sort=False):
            positions = [str(position) for position in team_data['starting_position'].tolist() if pd.notna(position)]
            if not positions:
                continue

            position_string = '-'.join(positions)
            mapped_formation = self.formation_mapping.get(position_string)
            is_new_pattern = position_string not in known_patterns

            base_row = team_data.iloc[0]
            row = {
                'game_id': base_row.get('game_id'),
                'team_id': base_row.get('team_id'),
                'season': base_row.get('season'),
                'competition_name': base_row.get('competition_name'),
                'competition_country': base_row.get('competition_country'),
                'formation': mapped_formation,
                'game_date': base_row.get('game_date'),
                'game_venue': base_row.get('game_venue'),
                'positions': position_string,
                'team_name': base_row.get('team_name'),
                'week': base_row.get('week'),
                'opponent_team_name': None,
                'opponent_team_id': None,
                'opponent_formation': None,
                'opponent_positions': None,
                'is_new_pattern': is_new_pattern,
            }
            formation_rows.append(row)

            if mapped_formation is None:
                unknown_patterns.append(
                    {
                        'game_id': int(base_row['game_id']),
                        'team_id': int(base_row['team_id']),
                        'team_name': base_row.get('team_name'),
                        'positions': position_string,
                        'is_new_pattern': is_new_pattern,
                    }
                )

        formations = pd.DataFrame(formation_rows)
        if formations.empty:
            return formations, unknown_patterns

        # Enrich each team row with the opponent's team and formation context from the same game.
        by_game: dict[Any, list[int]] = formations.groupby('game_id').indices
        for _, row_indexes in by_game.items():
            if len(row_indexes) != 2:
                continue

            left_idx, right_idx = list(row_indexes)
            left = formations.loc[left_idx]
            right = formations.loc[right_idx]

            formations.at[left_idx, 'opponent_team_name'] = right.get('team_name')
            formations.at[left_idx, 'opponent_team_id'] = right.get('team_id')
            formations.at[left_idx, 'opponent_formation'] = right.get('formation')
            formations.at[left_idx, 'opponent_positions'] = right.get('positions')

            formations.at[right_idx, 'opponent_team_name'] = left.get('team_name')
            formations.at[right_idx, 'opponent_team_id'] = left.get('team_id')
            formations.at[right_idx, 'opponent_formation'] = left.get('formation')
            formations.at[right_idx, 'opponent_positions'] = left.get('positions')

        return formations[self.config.game_formations], unknown_patterns

    def _upsert_lineups(self, lineups: pd.DataFrame) -> int:
        records = self._records_for_mongo(lineups)
        if not records:
            return 0

        ops = [
            UpdateOne(
                {
                    'season': record['season'],
                    'game_id': int(record['game_id']),
                    'team_id': int(record['team_id']),
                    'player_id': int(record['player_id']),
                },
                {'$set': record},
                upsert=True,
            )
            for record in records
            if record.get('team_id') is not None and record.get('player_id') is not None
        ]
        if not ops:
            return 0

        result = self.collection_lineups.bulk_write(ops, ordered=False)
        return int(result.upserted_count + result.modified_count)

    def _upsert_formations(self, formations: pd.DataFrame) -> int:
        records = self._records_for_mongo(formations)
        if not records:
            return 0

        ops = [
            UpdateOne(
                {
                    'season': record['season'],
                    'game_id': int(record['game_id']),
                    'team_id': int(record['team_id']),
                },
                {'$set': record},
                upsert=True,
            )
            for record in records
            if record.get('team_id') is not None
        ]
        if not ops:
            return 0

        result = self.collection_formations.bulk_write(ops, ordered=False)
        return int(result.upserted_count + result.modified_count)

    def save_new_finished_game_lineups(self, limit: Optional[int] = None) -> Dict[str, int]:
        finished_df = self._finished_games_df()
        if finished_df.empty:
            logger.info('No finished schedule games found for lineup extraction.')
            self.client.close()
            return {
                'processed_games': 0,
                'affected_rows': 0,
                'formation_rows': 0,
                'unmapped_formations': [],
                'new_unmapped_formations': [],
            }

        finished_df = finished_df.copy()
        finished_df['game_id'] = finished_df['game_id'].astype(int)
        self._patch_ws_schedule(finished_df)

        extracted_ids = self._lineups_extracted()
        game_ids = [int(x) for x in finished_df['game_id'].tolist()]
        pending = [gid for gid in game_ids if gid not in extracted_ids]
        if limit is not None:
            pending = pending[:limit]

        if not pending:
            logger.info('No new finished games to extract lineups for.')
            self.client.close()
            return {
                'processed_games': 0,
                'affected_rows': 0,
                'formation_rows': 0,
                'unmapped_formations': [],
                'new_unmapped_formations': [],
            }

        processed_games = 0
        affected_rows = 0
        formation_rows = 0
        unmapped_formations: list[dict[str, Any]] = []
        known_patterns = self._known_formation_patterns()

        for game_id in pending:
            try:
                logger.info('Fetching lineups for game_id=%s', game_id)
                loader = self.ws.read_events(
                    match_id=game_id,
                    output_fmt='loader',
                    retry_missing=True,
                    on_error='skip',
                )
                players = loader.players(game_id=game_id)
                if players is None or players.empty:
                    logger.info('No lineup players available for game_id=%s', game_id)
                    continue

                game_info = finished_df[finished_df['game_id'] == game_id].reset_index(drop=True)
                if game_info.empty:
                    logger.warning('Schedule metadata missing for game_id=%s', game_id)
                    continue

                lineups = self._process_game_info(players, game_info)
                formations, unknown_patterns = self._build_formations(lineups, known_patterns)
                affected_rows += self._upsert_lineups(lineups)
                formation_rows += self._upsert_formations(formations)
                known_patterns.update(formations['positions'].dropna().astype(str).tolist())
                unmapped_formations.extend(unknown_patterns)
                processed_games += 1
                logger.info('Saved %s lineup rows for game_id=%s', len(lineups), game_id)

            except Exception:
                logger.exception('Error processing lineups for game_id=%s', game_id)

        self.client.close()
        deduped_unmapped = list({
            (item['positions'], item['game_id'], item['team_id']): item
            for item in unmapped_formations
        }.values())
        new_unmapped = [item for item in deduped_unmapped if item.get('is_new_pattern')]
        return {
            'processed_games': processed_games,
            'affected_rows': affected_rows,
            'formation_rows': formation_rows,
            'unmapped_formations': deduped_unmapped,
            'new_unmapped_formations': new_unmapped,
        }


def main(config_path: Path | str = CONFIG_PATH) -> Dict[str, int]:
    setup_logging()
    config = load_app_config(str(config_path))
    lineups = GameLineups(config)
    return lineups.save_new_finished_game_lineups()


if __name__ == '__main__':
    main()
