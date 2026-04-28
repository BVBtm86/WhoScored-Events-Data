from __future__ import annotations

from pathlib import Path
from typing import Dict

from helper import setup_logging, patch_soccerdata, load_app_config, send_email, format_exception, logger
from game_schedule import GameSchedule
from raw_events import RawEvents
from backup_data import backup_season_data

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'


def build_report(schedule: Dict[str, int], raw_events: Dict[str, int], backup: Dict[str, int]) -> str:
    return (
        f"Schedule: processed={schedule.get('processed')} upserted={schedule.get('upserted')}\n"
        f"Raw events: processed_games={raw_events.get('processed_games')} inserted_rows={raw_events.get('inserted_rows')}\n"
        f"Backup: exported_count={backup.get('exported_count')}\n"
    )


def main(config_path: Path | str = CONFIG_PATH) -> None:
    setup_logging()
    patch_soccerdata()
    config = load_app_config(str(config_path))

    schedule_status = {'processed': 0, 'upserted': 0}
    raw_status = {'processed_games': 0, 'inserted_rows': 0}
    backup_status = {'exported_count': 0}
    report = ''
    status = 'SUCCESS'

    try:
        schedule_status = GameSchedule(config).save_schedule()
        raw_status = RawEvents(config).save_new_finished_games()
        if raw_status.get('inserted_rows') > 0:
            backup_status = backup_season_data(config)
        report = build_report(schedule_status, raw_status, backup_status)
        logger.info('Pipeline finished successfully.')
    except Exception as exc:
        status = 'FAILED'
        report = f"Pipeline failed: {format_exception(exc)}"
        logger.exception('Pipeline execution failed.')

    if config.email.password:
        subject = f"WhoScored pipeline {status} | {config.season.year}"
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
        logger.warning('Email not sent because SMTP password is missing.')


if __name__ == '__main__':
    main()
