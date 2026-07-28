from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from prod_pipeline.helper import setup_logging, patch_soccerdata, load_app_config, send_email, format_exception, logger
from prod_pipeline.game_schedule import GameSchedule
from prod_pipeline.game_lineups import GameLineups
from prod_pipeline.raw_events import RawEvents
from prod_pipeline.processed_events import ProcessedEvents
from prod_pipeline.game_stats import GameStats
from prod_pipeline.sequence_data import SequenceData
from prod_pipeline.backup_data import backup_season_data
import argparse

CONFIG_PATH = Path(__file__).resolve().parent / 'config' / 'config.yaml'

def build_report(
        schedule: Optional[Dict[str, int]] = None,
        lineups: Optional[Dict[str, int]] = None,
        raw_events: Optional[Dict[str, int]] = None,
        processed_events: Optional[Dict[str, int]] = None,
        game_stats: Optional[Dict[str, int]] = None,
        sequence_data: Optional[Dict[str, int]] = None,
        backup: Optional[Dict[str, int]] = None) -> str:
    schedule = schedule or {}
    lineups = lineups or {}
    raw_events = raw_events or {}
    processed_events = processed_events or {}
    game_stats = game_stats or {}
    sequence_data = sequence_data or {}
    backup = backup or {}
    return (
        f"Schedule: processed={schedule.get('processed')} upserted={schedule.get('upserted')}\n"
        f"Lineups: processed_games={lineups.get('processed_games')} affected_rows={lineups.get('affected_rows')} formation_rows={lineups.get('formation_rows')}\n"
        f"Raw events: processed_games={raw_events.get('processed_games')} inserted_rows={raw_events.get('inserted_rows')}\n"
        f"Processed events: processed_games={processed_events.get('processed_games')} inserted_rows={processed_events.get('inserted_rows')}\n"
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


def main(config_path: Path | str = CONFIG_PATH, season_year: str | None = None) -> None:
    setup_logging()
    patch_soccerdata()
    config = load_app_config(str(config_path))

    if season_year:
        config.season.year = season_year
        
    schedule_status = {'processed': 0, 'upserted': 0}
    lineups_status = {'processed_games': 0, 'affected_rows': 0}
    raw_status = {'processed_games': 0, 'inserted_rows': 0}
    processed_status = {'processed_games': 0, 'inserted_rows': 0}
    game_stats_status = {'processed_games': 0, 'team_rows': 0, 'player_rows': 0}
    sequence_status = {'processed_games': 0, 'shot_rows': 0, 'pass_rows': 0}
    backup_status = {'exported_count': 0}
    report = ''
    status = 'SUCCESS'

    try:
        # schedule_status = GameSchedule(config).save_schedule()
        lineups_status = GameLineups(config).save_new_finished_game_lineups()
        # raw_status = RawEvents(config).save_new_finished_games()
        # processed_status = ProcessedEvents(config).save_processed_raw_games()
        game_stats_status = GameStats(config).save_game_stats()
        sequence_status = SequenceData(config).save_sequence_data()
        if (
            lineups_status.get('affected_rows', 0) > 0
            or raw_status.get('inserted_rows', 0) > 0
            or processed_status.get('inserted_rows', 0) > 0
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
            game_stats=game_stats_status,
            sequence_data=sequence_status,
            backup=backup_status,
        )
        logger.info('Pipeline finished successfully.')
    except Exception as exc:
        status = 'FAILED'
        report = f"Pipeline failed: {format_exception(exc)}"
        logger.exception('Pipeline execution failed.')

    if config.email and config.email.password:
        day = datetime.now().date().strftime('%d-%m-%Y')
        subject = f"{day} WhoScored pipeline {status} | {config.season.year}"
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
    else:
        logger.warning('Email not sent because SMTP password is missing or email is not configured.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the WhoScored data pipeline.')
    parser.add_argument(
        '--season-year',
        help='Override season.year from config.yaml, for example 2009-2010.',
    )

    args = parser.parse_args()
    main(season_year=args.season_year)
