from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from prod_pipeline.helper import setup_logging, patch_soccerdata, load_app_config, send_email, format_exception, logger
from prod_pipeline.game_schedule import GameSchedule, refresh_schedule_details
from prod_pipeline.game_lineups import GameLineups
from prod_pipeline.raw_events import RawEvents
from prod_pipeline.processed_events import ProcessedEvents
from prod_pipeline.process_game_key_events import KeyEventProcessor
from prod_pipeline.game_stats import GameStats
from prod_pipeline.sequence_data import SequenceData
from prod_pipeline.backup_data import backup_season_data
from pymongo import MongoClient
import argparse

CONFIG_PATH = Path(__file__).resolve().parent / 'config' / 'config.yaml'


def clear_derived_season(config) -> None:
    """Remove only derived documents for a failed season; preserve imported sources."""
    client = MongoClient(config.mongo.url, serverSelectionTimeoutMS=5000)
    try:
        db = client[config.mongo.db]
        season = config.season.year
        game_ids = set(db[config.mongo.collections['collection_raw_events']].distinct(
            'game_id', {'season': season}
        ))
        derived = [
            'collection_processed_events', 'collection_team_game_stats',
            'collection_player_game_stats', 'collection_shot_sequences',
            'collection_pass_sequences',
        ]
        for key in derived:
            name = config.mongo.collections.get(key)
            if name:
                result = db[name].delete_many({'season': season})
                logger.warning('Failure cleanup: deleted %s documents from %s', result.deleted_count, name)
        key_events = db['game_key_events'].delete_many({'season': season})
        logger.warning('Failure cleanup: deleted %s documents from game_key_events', key_events.deleted_count)
        if game_ids:
            state = db['game_key_event_state'].delete_many({'game_id': {'$in': list(game_ids)}})
            logger.warning('Failure cleanup: deleted %s documents from game_key_event_state', state.deleted_count)
    finally:
        client.close()

def build_report(
        schedule: Optional[Dict[str, int]] = None,
        lineups: Optional[Dict[str, int]] = None,
        raw_events: Optional[Dict[str, int]] = None,
        processed_events: Optional[Dict[str, int]] = None,
        key_events: Optional[Dict[str, int]] = None,
        game_stats: Optional[Dict[str, int]] = None,
        sequence_data: Optional[Dict[str, int]] = None,
        backup: Optional[Dict[str, int]] = None) -> str:
    schedule = schedule or {}
    lineups = lineups or {}
    raw_events = raw_events or {}
    processed_events = processed_events or {}
    key_events = key_events or {}
    game_stats = game_stats or {}
    sequence_data = sequence_data or {}
    backup = backup or {}
    return (
        f"Schedule: processed={schedule.get('processed')} upserted={schedule.get('upserted')}\n"
        f"Lineups: processed_games={lineups.get('processed_games')} affected_rows={lineups.get('affected_rows')} formation_rows={lineups.get('formation_rows')}\n"
        f"Lineup failures: {lineups.get('failed_games', 0)}\n"
        f"Raw events: processed_games={raw_events.get('processed_games')} inserted_rows={raw_events.get('inserted_rows')}\n"
        f"Processed events: processed_games={processed_events.get('processed_games')} inserted_rows={processed_events.get('inserted_rows')}\n"
        f"Key events: processed={key_events.get('processed')} skipped={key_events.get('skipped')} events={key_events.get('events')} failed={key_events.get('failed')} ambiguous={key_events.get('ambiguous')}\n"
        f"Game stats: processed_games={game_stats.get('processed_games')} team_rows={game_stats.get('team_rows')} player_rows={game_stats.get('player_rows')}\n"
        f"Sequence data: processed_games={sequence_data.get('processed_games')} shot_rows={sequence_data.get('shot_rows')} pass_rows={sequence_data.get('pass_rows')}\n"
        f"Backup: exported_count={backup.get('exported_count')}\n"
        + (
            "New formation patterns:\n"
            + "\n".join(
                f"- game_id={item.get('game_id')} team={item.get('team_name')} positions={item.get('positions')}"
                for item in lineups.get('new_unmapped_formations', [])
            )
            + "\n"
            if lineups.get('new_unmapped_formations')
            else ""
        )
    )


def main(
        config_path: Path | str = CONFIG_PATH,
        season_year: str | None = None,
        full: bool = False,
        mongo_uri: str | None = None,
        schedule_only: bool = False,
        matchday: int | None = None,
        rebuild_derived: bool = False) -> None:
    if matchday is not None and (not schedule_only or not 1 <= matchday <= 34):
        raise ValueError('--matchday requires --schedule-only and a value from 1 to 34.')
    setup_logging()
    patch_soccerdata()
    config = load_app_config(str(config_path))
    if mongo_uri:
        config.mongo.url = mongo_uri

    if season_year:
        config.season.year = season_year
        start_year, end_year = season_year.split('-')
        config.season.year_short = start_year[-2:] + end_year[-2:]

    if schedule_only:
        result = GameSchedule(config, force_cache=False).save_schedule(matchday=matchday)
        logger.info('Schedule-only run finished: %s. Other stages and email skipped.', result)
        return
        
    schedule_status = {'processed': 0, 'upserted': 0}
    lineups_status = {'processed_games': 0, 'affected_rows': 0}
    raw_status = {'processed_games': 0, 'inserted_rows': 0}
    processed_status = {'processed_games': 0, 'inserted_rows': 0}
    key_events_status = {'processed': 0, 'skipped': 0, 'events': 0, 'failed': 0, 'ambiguous': 0}
    game_stats_status = {'processed_games': 0, 'team_rows': 0, 'player_rows': 0}
    sequence_status = {'processed_games': 0, 'shot_rows': 0, 'pass_rows': 0}
    backup_status = {'exported_count': 0}
    report = ''
    status = 'SUCCESS'

    try:
        if full and not rebuild_derived:
            schedule_status = GameSchedule(config, force_cache=False).save_schedule()
        if not rebuild_derived:
            lineups_status = GameLineups(config).save_new_finished_game_lineups()
        if full and not rebuild_derived:
            raw_status = RawEvents(config).save_new_finished_games()
        if not rebuild_derived:
            refresh_schedule_details(config)
        processed_status = ProcessedEvents(config).save_processed_raw_games()
        key_events_status = KeyEventProcessor(
            config.mongo.url,
            config.mongo.db,
        ).run("backfill", season=config.season.year, stop_on_error=True)
        game_stats_status = GameStats(config).save_game_stats()
        sequence_status = SequenceData(config).save_sequence_data()
        if (
            lineups_status.get('affected_rows', 0) > 0
            or raw_status.get('inserted_rows', 0) > 0
            or processed_status.get('inserted_rows', 0) > 0
            or key_events_status.get('events', 0) > 0
            or game_stats_status.get('team_rows', 0) > 0
            or game_stats_status.get('player_rows', 0) > 0
            or sequence_status.get('shot_rows', 0) > 0
            or sequence_status.get('pass_rows', 0) > 0
        ):
            backup_status = backup_season_data(config)

        report = build_report(
            schedule=schedule_status,
            lineups=lineups_status,
            raw_events=raw_status,
            processed_events=processed_status,
            key_events=key_events_status,
            game_stats=game_stats_status,
            sequence_data=sequence_status,
            backup=backup_status,
        )
        if lineups_status.get('failed_games', 0):
            status = 'PARTIAL'
            logger.warning('Pipeline finished with %s lineup failures.', lineups_status['failed_games'])
        else:
            logger.info('Pipeline finished successfully.')
    except Exception as exc:
        status = 'FAILED'
        report = f"Pipeline failed: {format_exception(exc)}"
        logger.exception('Pipeline execution failed.')
        try:
            clear_derived_season(config)
        except Exception:
            logger.exception('Failure cleanup also failed for season %s.', config.season.year)
        raise

    if config.email and config.email.password:
        day = datetime.now().date().strftime('%d-%m-%Y')
        subject = f"{day} WhoScored pipeline {status} | {config.season.year}"
        try:
            send_email(
                smtp_host=config.email.smtp_host,
                smtp_port=config.email.smtp_port,
                username=config.email.username,
                password=config.email.password,
                from_email=config.email.from_email,
                to_email=config.email.to_email,
                subject=subject,
                body=report,
                use_tls=config.email.use_tls,
            )
        except Exception:
            logger.warning('Email notification failed. Pipeline status: %s.', status)
    else:
        logger.warning('Email not sent because SMTP password is missing or email is not configured.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the WhoScored data pipeline.')
    parser.add_argument(
        '--season-year',
        help='Override season.year from config.yaml, for example 2009-2010.',
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Collect schedule and raw events, process events, then build stats and sequences.',
    )
    parser.add_argument(
        '--rebuild-derived',
        action='store_true',
        help='Rebuild processed events, key events, stats and sequences from imported source collections without scraping.',
    )
    parser.add_argument(
        '--mongo-uri',
        help='Override the MongoDB connection for this run.',
    )
    parser.add_argument(
        '--schedule-only', action='store_true',
        help='Save all fixtures to MongoDB; skip other stages and email.',
    )
    parser.add_argument(
        '--matchday', type=int, choices=range(1, 35), metavar='1-34',
        help='Save only this matchday. Requires --schedule-only.',
    )

    args = parser.parse_args()
    if args.matchday is not None and not args.schedule_only:
        parser.error('--matchday requires --schedule-only')
    main(season_year=args.season_year, full=args.full, mongo_uri=args.mongo_uri,
         schedule_only=args.schedule_only, matchday=args.matchday,
         rebuild_derived=args.rebuild_derived)
