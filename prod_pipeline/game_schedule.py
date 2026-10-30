from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
import soccerdata as sd
from pymongo import MongoClient, UpdateOne
from soccerdata.whoscored import WHOSCORED_DATADIR

from prod_pipeline.helper import setup_logging, patch_soccerdata, load_app_config, logger
from prod_pipeline.config import ScrapeDataConfig

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'


def optional_value(value):
    """Return scalar source metadata, converting pandas missing values to None."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def match_details(game_id: int, data: dict) -> dict:
    return {
        'game_id': game_id,
        'stats_available': bool(data.get('events')),
        'home_manager': (data.get('home') or {}).get('managerName'),
        'away_manager': (data.get('away') or {}).get('managerName'),
        'referee': (data.get('referee') or {}).get('name'),
        'venue': data.get('venueName'),
        'attendance': data.get('attendance'),
    }


def refresh_schedule_details(config: ScrapeDataConfig) -> Dict[str, int]:
    """Refresh finished-match metadata after downloads, without starting a browser."""
    folder = WHOSCORED_DATADIR / 'events' / f'{config.season.league}_{config.season.year_short}'
    with MongoClient(config.mongo.url) as client:
        collection = client[config.mongo.db][config.mongo.collections['collection_schedule']]
        collection.update_many({'season': config.season.year}, {'$unset': {'source_status': ''}})
        operations = []
        for row in collection.find({'season': config.season.year, 'game_status': 'finished'}, {'game_id': 1}):
            game_id = int(row['game_id'])
            path = folder / f'{game_id}.json'
            if not path.exists():
                continue
            with path.open('rb') as stream:
                data = json.load(stream)
            if not data:
                continue
            details = {
                key: optional_value(value) for key, value in match_details(game_id, data).items()
                if optional_value(value) is not None
            }
            operations.append(UpdateOne({'season': config.season.year, 'game_id': game_id}, {'$set': details}))
        if operations:
            collection.bulk_write(operations, ordered=False)
    logger.info('Refreshed cached match details for %s finished fixtures.', len(operations))
    return {'processed': len(operations)}


class GameSchedule:
    def __init__(self, config: ScrapeDataConfig, force_cache: bool = True):
        self.config = config
        self.force_cache = force_cache
        patch_soccerdata()
        self.ws = sd.WhoScored(
            leagues=config.season.league,
            seasons=config.season.year,
            no_cache=not force_cache,
        )
        self.fbref = sd.FBref(
            leagues=config.season.league,
            seasons=config.season.year,
            no_cache=not force_cache,
        )
        self.client = MongoClient(config.mongo.url)
        self.db = self.client[config.mongo.db]
        self.collection_schedule = self.db[config.mongo.collections['collection_schedule']]
        self.collection_teams = self.db[config.mongo.collections['collection_teams']]

    def _read_team_mapping(self) -> dict[int, str]:
        cursor = self.collection_teams.find({}, {"_id": 0, "ws_team_id": 1, "fbref_team_name": 1})
        return {int(doc["ws_team_id"]): doc["fbref_team_name"] for doc in cursor}

    def _read_season_schedule(self) -> pd.DataFrame:
        df = self.ws.read_schedule(force_cache=self.force_cache)
        df = df.reset_index(drop=False)
        if 'start_time' not in df.columns and 'date' in df.columns:
            df = df.rename(columns={'date': 'start_time'})
        return df

    def _read_fbref_schedule(self) -> pd.DataFrame:
        df = self.fbref.read_schedule().reset_index(drop=False)
        for column in ('home_team', 'away_team'):
            df[column] = df[column].replace({'Frankfurt': 'Eintracht Frankfurt'})
            # Keep schedule join keys consistent even when FBref returns a
            # missing/non-string team value for an unfinished fixture.
            df[column] = df[column].astype('string').str.strip()
        df["fbref_date"] = pd.to_datetime(df["date"]).dt.date
        return df[["week", "fbref_date", "home_team", "away_team"]].rename(columns={
            "home_team": "fbref_home_team", "away_team": "fbref_away_team", "week": "week",
        })

    def _process_fbref_schedule(self, df_ws: pd.DataFrame) -> pd.DataFrame:
        team_map = self._read_team_mapping()
        df_ws["match_date"] = pd.to_datetime(df_ws["start_time"]).dt.date
        df_ws["fbref_home_team"] = df_ws["home_team_id"].astype(int).map(team_map)
        df_ws["fbref_away_team"] = df_ws["away_team_id"].astype(int).map(team_map)
        # ``Series.map`` produces float64 when all values are missing.  FBref
        # keys are strings, so normalize both sides explicitly before merge.
        # Nullable StringDtype preserves unmatched teams as <NA> without
        # turning them into the literal string "nan".
        for column in ("fbref_home_team", "fbref_away_team"):
            df_ws[column] = df_ws[column].astype('string').str.strip()
        df_fbref = self._read_fbref_schedule()
        df = pd.merge(left=df_ws, right=df_fbref,
                      left_on=["match_date", "fbref_home_team", "fbref_away_team"],
                      right_on=["fbref_date", "fbref_home_team", "fbref_away_team"],
                      how="left", validate="many_to_one")
        pairing_columns = ['fbref_home_team', 'fbref_away_team']
        unique_pairs = df_fbref.dropna(subset=pairing_columns + ['week']).drop_duplicates(
            subset=pairing_columns, keep=False,
        ).set_index(pairing_columns)['week'].to_dict()
        missing = df['week'].isna()
        df.loc[missing, 'week'] = [
            unique_pairs.get((home, away))
            for home, away in df.loc[missing, pairing_columns].itertuples(index=False, name=None)
        ]
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

            rows.append(match_details(game_id, data))

        info_columns = [*self.config.schedule_games.match_stats_info, "stats_available"]
        season_games_info = pd.DataFrame(rows, columns=info_columns)

        return season_games_info
    
    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        required = set(self.config.schedule_games.required_columns)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"WhoScored schedule is missing required columns: {sorted(missing)}")

    def _prepare_schedule(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_required_columns(df)
        finished = df.copy()
        if finished.empty:
            return finished

        missing_week = pd.to_numeric(finished['week'], errors='coerce').isna()
        if missing_week.any():
            logger.warning(
                'No FBref matchday for %s fixtures (game_ids=%s). '
                'Saving them with week=null so event collection can continue.',
                int(missing_week.sum()), finished.loc[missing_week, 'game_id'].tolist(),
            )

        if 'date' in finished.columns or 'start_time' in finished.columns:
            # SoccerData's date comes from startTimeUtc; start_time is local.
            date_column = 'date' if 'date' in finished.columns else 'start_time'
            finished['game_date'] = pd.to_datetime(finished[date_column], errors='coerce', utc=True)
        else:
            finished['game_date'] = pd.NaT

        return finished

    def _prepare_finished_schedule(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._prepare_schedule(
            df[df['status'] == self.config.schedule_games.finished_status_code]
        )

    def _build_docs(self, df: pd.DataFrame) -> list[dict]:
        docs: list[dict] = []
        for row in df.to_dict(orient='records'):
            game_id = int(row["game_id"])
            start_time = row.get('game_date')
            week = row.get('week')
            week = int(week) if pd.notna(week) else None
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

            is_finished = row.get('status') == self.config.schedule_games.finished_status_code

            doc = {
                "game_id": game_id,
                "game_date": start_time,
                "season": self.config.season.year,
                "week": week,
                "competition_name": self.config.season.name,
                "competition_country": self.config.season.country,
                "home_team_id": row.get("home_team_id"),
                "home_team_name": row.get("home_team"),
                "home_goals": row.get("home_score") if is_finished else None,
                "away_team_id": row.get("away_team_id"),
                "away_team_name": row.get("away_team"),
                "away_goals": row.get("away_score") if is_finished else None,
                "game_status": "finished" if is_finished else "scheduled",
                "stats_available": stats_available,
            }
            # Do not write pandas NaN over valid metadata already stored for a
            # fixture.  Metadata is allowed to arrive later from event files.
            for field in ('home_manager', 'away_manager', 'referee', 'venue', 'attendance'):
                value = optional_value(row.get(field))
                if value is not None:
                    doc[field] = value
            docs.append(doc)
        return docs

    def save_schedule(self, matchday: int | None = None) -> Dict[str, int]:
        if matchday is not None and not 1 <= matchday <= 34:
            raise ValueError('Matchday must be between 1 and 34.')
        try:
            return self._save_schedule(matchday)
        finally:
            self.client.close()
            for reader in (self.ws, self.fbref):
                driver = getattr(reader, '_driver', None)
                if driver is not None:
                    try:
                        driver.quit()
                    except Exception:
                        logger.warning('Could not close scraper browser.', exc_info=True)

    def _save_schedule(self, matchday: int | None) -> Dict[str, int]:
        
        df_ws = self._read_season_schedule()
        schedule_df = self._process_fbref_schedule(df_ws=df_ws)

        finished_df = self._prepare_schedule(df=schedule_df)
        if matchday is not None:
            finished_df = finished_df[
                pd.to_numeric(finished_df['week'], errors='coerce') == matchday
            ].copy()
            logger.info('Restricted schedule import to matchday %s.', matchday)

        if finished_df.empty:
            logger.info('No fixtures found in schedule.')
            return {'processed': 0, 'upserted': 0}

        game_ids = finished_df.loc[
            finished_df['status'] == self.config.schedule_games.finished_status_code,
            'game_id',
        ].dropna().astype(int).tolist()
        games_info = self._extract_games_info(game_ids)

        finished_df = pd.merge(
            left=finished_df,
            right=games_info,
            on="game_id",
            how="left",
        )

        docs = self._build_docs(finished_df)
        ops = [
            UpdateOne({'game_id': doc['game_id']}, {'$set': doc, '$unset': {'source_status': ''}}, upsert=True)
            for doc in docs
        ]

        result = self.collection_schedule.bulk_write(ops, ordered=False)
        affected = int(result.upserted_count + result.modified_count)
        logger.info('Schedule saved: %s fixtures (%s finished), %s affected docs', len(docs), len(game_ids), affected)
        return {'processed': len(docs), 'upserted': affected}


def main(config_path: Path | str = CONFIG_PATH) -> Dict[str, int]:
    setup_logging()
    config = load_app_config(str(config_path))
    schedule = GameSchedule(config)
    return schedule.save_schedule()


if __name__ == '__main__':
    main()
