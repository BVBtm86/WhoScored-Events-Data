from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
import soccerdata as sd
from pymongo import MongoClient, UpdateOne

from prod_pipeline.helper import setup_logging, patch_soccerdata, load_app_config, logger
from prod_pipeline.config import ScrapeDataConfig

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'


class GameSchedule:
    def __init__(self, config: ScrapeDataConfig):
        self.config = config
        patch_soccerdata()
        self.ws = sd.WhoScored(
            leagues=config.season.league,
            seasons=config.season.year,
        )
        self.fbref = sd.FBref(
            leagues=config.season.league,
            seasons=config.season.year,
        )

        self.client = MongoClient(config.mongo.url)
        self.db = self.client[config.mongo.db]
        self.collection_schedule = self.db[config.mongo.collections['collection_schedule']]
        self.collection_teams = self.db[config.mongo.collections['collection_teams']]

    def _read_team_mapping(self) -> dict[int, str]:
        cursor = self.collection_teams.find(
            {},
            {"_id": 0, "ws_team_id": 1, "fbref_team_name": 1},
        )

        return {
            int(doc["ws_team_id"]): doc["fbref_team_name"]
            for doc in cursor
        }

    def _read_season_schedule(self) -> pd.DataFrame:
        df = self.ws.read_schedule(force_cache=True)
        return df.reset_index(drop=False)

    def _read_fbref_schedule(self) -> pd.DataFrame:

        df = self.fbref.read_schedule().reset_index(drop=False)

        df["fbref_date"] = pd.to_datetime(df["date"]).dt.date

        return df[["week", "fbref_date", "home_team", "away_team"]].rename(
            columns={
                "home_team": "fbref_home_team",
                "away_team": "fbref_away_team",
                "week": "week",
            }
        )
    
    def _process_fbref_schedule(self,
                                df_ws: pd.DataFrame) -> pd.DataFrame:
    
        team_map = self._read_team_mapping()

        df_ws["match_date"] = pd.to_datetime(df_ws["start_time"]).dt.date
        df_ws["fbref_home_team"] = df_ws["home_team_id"].astype(int).map(team_map)
        df_ws["fbref_away_team"] = df_ws["away_team_id"].astype(int).map(team_map)

        df_fbref = self._read_fbref_schedule()

        df = pd.merge(
            left=df_ws,
            right=df_fbref,
            left_on=["match_date", "fbref_home_team", "fbref_away_team"],
            right_on=["fbref_date", "fbref_home_team", "fbref_away_team"],
            how="left",
        )

        return df

    def _extract_games_info(self, game_ids: list[int]) -> pd.DataFrame:

        rows: list[dict] = []
        events_dir = (
            self.ws.data_dir
            / "events"
            / f"{self.config.season.league}_{self.config.season.year_short}"
        )

        for game_id in game_ids:
            event_file = events_dir / f"{game_id}.json"
            if not event_file.exists():
                logger.warning("No cached WhoScored event file for game_id=%s", game_id)
                continue

            with event_file.open("rb") as f:
                data = json.load(f)

            if not data:
                logger.warning("Skipping empty WhoScored event file for game_id=%s", game_id)
                continue

            referee = data.get("referee") or {}
            rows.append(
                {
                    "game_id": game_id,
                    "stats_available": True,
                    "home_manager": (data.get("home") or {}).get("managerName"),
                    "away_manager": (data.get("away") or {}).get("managerName"),
                    "referee": referee.get("name"),
                    "venue": data.get("venueName"),
                    "attendance": data.get("attendance"),
                }
            )

        info_columns = [*self.config.schedule_games.match_stats_info, "stats_available"]
        season_games_info = pd.DataFrame(rows, columns=info_columns)

        return season_games_info
    
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
            game_id = int(row["game_id"])
            start_time = row.get('game_date')
            week = int(row.get("week"))
            if pd.notna(start_time):
                if hasattr(start_time, 'to_pydatetime'):
                    start_time = start_time.to_pydatetime()
            else:
                start_time = None

            stats_available = row.get("stats_available")
            if pd.isna(stats_available):
                stats_available = False
            else:
                stats_available = bool(stats_available)

            docs.append({
                "game_id": game_id,
                "game_date": start_time,
                "season": self.config.season.year,
                "week": week,
                "competition_name": self.config.season.name,
                "competition_country": self.config.season.country,
                "home_team_id": row.get("home_team_id"),
                "home_team_name": row.get("home_team"),
                "home_goals": row.get("home_score"),
                "home_manager": row.get("home_manager"),
                "away_team_id": row.get("away_team_id"),
                "away_team_name": row.get("away_team"),
                "away_goals": row.get("away_score"),
                "away_manager": row.get("away_manager"),
                "game_status": "finished",
                "stats_available": stats_available,
                'referee': row.get('referee'),
                'venue': row.get('venue'),
                'attendance': row.get('attendance')
            })
        return docs

    def save_schedule(self) -> Dict[str, int]:
        
        df_ws = self._read_season_schedule()
        schedule_df = self._process_fbref_schedule(df_ws=df_ws)

        finished_df = self._prepare_finished_schedule(df=schedule_df)

        if finished_df.empty:
            logger.info('No finished games found in schedule.')
            self.client.close()
            return {'processed': 0, 'upserted': 0}

        game_ids = finished_df["game_id"].dropna().astype(int).tolist()
        games_info = self._extract_games_info(game_ids)

        finished_df = pd.merge(
            left=finished_df,
            right=games_info,
            on="game_id",
            how="left",
        )

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
