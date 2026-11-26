from __future__ import annotations

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    env_path_file = project_root / ".venv-path"
    project_env = (
        Path(env_path_file.read_text().strip())
        if env_path_file.is_file()
        else project_root / ".venv"
    )
    project_python = project_env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if project_python.is_file() and Path(sys.prefix).resolve() != project_env.resolve():
        os.execv(str(project_python), [str(project_python), "-Bu", *sys.argv])

from typing import Dict, Optional
from datetime import datetime, timezone
from html import escape
from prod_pipeline.helper import setup_logging, patch_soccerdata, load_app_config, send_email, format_exception, logger
from prod_pipeline.indexes import ensure_application_indexes
from pymongo import MongoClient
import argparse

CONFIG_PATH = Path(__file__).resolve().parent / 'config' / 'config.yaml'


def atlas_connection_uri() -> str:
    uri = os.environ.get("ATLAS_MONGO_URI", "").strip()
    if uri:
        return uri

    env_file = CONFIG_PATH.parent.parent / ".env"
    if not env_file.is_file():
        return ""
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.strip().removeprefix("export ").partition("=")
        if separator and key.strip() == "ATLAS_MONGO_URI":
            value = value.strip()
            if len(value) >= 2 and value[0] in ("'", '"') and value[-1] == value[0]:
                return value[1:-1]
            return value
    return ""



def matchday_coverage(database, config, attempted_ids):
    season = {"season": config.season.year}
    collections = config.mongo.collections
    schedule = database[collections["collection_schedule"]]
    fixtures = list(schedule.find(season, {
        "_id": 0, "game_id": 1, "week": 1, "game_date": 1, "game_status": 1,
        "home_team_name": 1, "away_team_name": 1,
    }).sort([("game_date", 1), ("game_id", 1)]))
    weeks = {game.get("week") for game in fixtures if game["game_id"] in attempted_ids}
    latest = None
    if not weeks:
        latest = next((game for game in reversed(fixtures)
                       if game.get("game_date") and game["game_date"] <= datetime.now(timezone.utc).replace(tzinfo=None)), None)
        if latest:
            weeks.add(latest.get("week"))

    selected = [
        game for game in fixtures
        if game.get("week") in weeks
        and (game.get("week") is not None or game["game_id"] in attempted_ids or game is latest)
    ]
    counts = {}
    if selected:
        raw = database[collections["collection_raw_events"]]
        counts = {row["_id"]: row["count"] for row in raw.aggregate([
            {"$match": {**season, "game_id": {"$in": [game["game_id"] for game in selected]}}},
            {"$group": {"_id": "$game_id", "count": {"$sum": 1}}},
        ])}
    return [
        (f"Matchday {week}" if week is not None else "Unassigned matchday",
         [(game, counts.get(game["game_id"], 0)) for game in selected if game.get("week") == week])
        for week in sorted(weeks, key=lambda week: (week is None, week or 0))
    ]


def fixture_collection_counts(database, season, coverage, collection_names):
    game_ids = [
        game["game_id"] for _, fixtures in coverage for game, _ in fixtures
        if game.get("game_status") == "finished"
    ]
    counts = {game_id: {} for game_id in game_ids}
    if not game_ids:
        return counts

    for name in collection_names:
        rows = database[name].aggregate([
            {"$match": {"season": season, "game_id": {"$in": game_ids}}},
            {"$group": {"_id": "$game_id", "count": {"$sum": 1}}},
        ])
        for row in rows:
            counts[row["_id"]][name] = row["count"]
    return counts


def report_table(headers, rows):
    plain = [" | ".join(headers), *(" | ".join(row) for row in rows)]
    headings = "".join(
        f'<th scope="col" style="padding:9px;background:#eaf0f6;text-align:left">{escape(cell)}</th>'
        for cell in headers
    )
    body = "".join("<tr>" + "".join(
        f'<td style="padding:9px;border-bottom:1px solid #dce4ed">{escape(cell)}</td>' for cell in row
    ) + "</tr>" for row in rows)
    html = (
        '<table style="width:100%;border-collapse:collapse;margin-bottom:20px">'
        f'<thead><tr>{headings}</tr></thead><tbody>{body}</tbody></table>'
    )
    return plain, html


def report_disclosure(label, *, open_by_default=False, group=None):
    open_attr = " open" if open_by_default else ""
    group_attr = f' name="{escape(group)}"' if group else ""
    return (
        f'<details{open_attr}{group_attr} style="margin:0 0 16px">'
        '<summary style="cursor:pointer;background:#eaf0f6;padding:10px;'
        f'font-size:16px;font-weight:bold;color:#172f48">{escape(label)}</summary>'
    )


def format_fixture_details(coverage, collection_names, counts):
    plain = ["Fixture details"]
    html = ['<section id="fixture-details"><h2 style="font-size:18px">Fixture details</h2>']
    for matchday, fixtures in coverage:
        plain.extend(["", matchday])
        html.append(report_disclosure(matchday))
        for game, _ in fixtures:
            fixture = f"{game.get('home_team_name') or 'Home'} vs {game.get('away_team_name') or 'Away'}"
            plain.extend(["", fixture])
            html.append(report_disclosure(fixture, group="report-fixtures"))
            if game.get("game_status") != "finished":
                plain.append("Awaiting result")
                html.append("<p>Awaiting result</p></details>")
                continue

            game_counts = counts.get(game["game_id"], {})
            rows = []
            for name in collection_names:
                count = game_counts.get(name, 0)
                rows.append((name, f"{count:,}", "✓ Available" if count else "Missing"))
            table_text, table_html = report_table(("Collection", "Documents", "Coverage"), rows)
            plain.extend(table_text)
            html.extend([table_html, "</details>"])
        html.append("</details>")
    html.append("</section>")
    return "\n".join(plain), "".join(html)


def format_email(season, status, coverage, local_counts, atlas_counts):
    title = f"Bundesliga data report | {season} | {status}"
    plain = [title]
    html = [f'<h1 style="font-size:24px;color:#172f48;margin:0 0 16px">{escape(title)}</h1>']

    def table(headers, rows):
        table_text, table_html = report_table(headers, rows)
        plain.extend(table_text)
        html.append(table_html)

    if not coverage:
        plain.append("No matchday coverage available.")
        html.append("<p>No matchday coverage available.</p>")
    for matchday, fixtures in coverage:
        finished = sum(game.get("game_status") == "finished" for game, _ in fixtures)
        missing = sum(game.get("game_status") == "finished" and count == 0 for game, count in fixtures)
        summary = (f"{len(fixtures)} fixtures · {finished} finished · "
                   f"{len(fixtures) - finished} awaiting result · {missing} finished without raw events")
        plain.extend(["", matchday, summary])
        html.append(report_disclosure(matchday, open_by_default=True))
        html.append(f"<p>{escape(summary)}</p>")
        rows = []
        for game, count in fixtures:
            fixture = f"{game.get('home_team_name') or 'Home'} vs {game.get('away_team_name') or 'Away'}"
            if game.get("game_status") == "finished":
                rows.append((fixture, f"{count:,}", "✓ Available" if count else "Missing raw events"))
            else:
                rows.append((fixture, "—", "Awaiting result"))
        table(("Fixture", "Raw events stored", "Coverage"), rows)
        html.append("</details>")

    plain.extend(["", "Collection updates"])
    html.append('<h2 style="font-size:18px">Collection updates</h2>')
    rows = [
        ("Local MongoDB", name, f"✓ Wrote {count:,}" if count else
         "No documents written" if count == 0 else "No refresh tracked")
        for name, count in local_counts.items()
    ]
    rows.extend(
        ("Atlas", name,
         "Run changes unavailable" if count is None else
         f"✓ Added {count.added:,} · Updated {count.updated:,}" if count.total else "No changes")
        for name, count in atlas_counts.items()
    )
    table(("Database", "Collection", "Run result"), rows)
    return "\n".join(plain), '<div style="max-width:760px;font:14px Arial,sans-serif;color:#223246">' + "".join(html) + "</div>"


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
        rebuild_derived: bool = False,
        atlas_only: bool = False) -> None:
    if matchday is not None and (not schedule_only or not 1 <= matchday <= 34):
        raise ValueError('--matchday requires --schedule-only and a value from 1 to 34.')
    setup_logging()
    config = load_app_config(str(config_path))
    if mongo_uri:
        config.mongo.url = mongo_uri

    if season_year:
        config.season.year = season_year
        start_year, end_year = season_year.split('-')
        config.season.year_short = start_year[-2:] + end_year[-2:]


    atlas_uri = atlas_connection_uri()
    if not atlas_uri and not schedule_only:
        raise ValueError("Set ATLAS_MONGO_URI in the environment or project .env to publish app collections to Atlas.")

    def sync_atlas(*, fixtures_only: bool = False):
        if atlas_uri:
            from prod_pipeline.atlas_sync import publish_season

            counts = publish_season(config, atlas_uri, schedule_only=fixtures_only)
            logger.info("Atlas publication complete for %s: %s", config.season.year, counts)
            return counts
        return {}

    if atlas_only:
        sync_atlas()
        return

    patch_soccerdata()
    from prod_pipeline.game_schedule import GameSchedule, refresh_schedule_details
    from prod_pipeline.game_lineups import GameLineups
    from prod_pipeline.raw_events import RawEvents
    from prod_pipeline.processed_events import ProcessedEvents
    from prod_pipeline.process_game_key_events import KeyEventProcessor
    from prod_pipeline.game_stats import GameStats
    from prod_pipeline.sequence_data import SequenceData
    from prod_pipeline.atlas_sync import publish_app_projections
    from prod_pipeline.backup_data import backup_season_data, missing_app_backups

    if schedule_only:
        result = GameSchedule(config, force_cache=False).save_schedule(matchday=matchday)
        sync_atlas(fixtures_only=True)
        logger.info('Schedule-only run finished: %s. Other stages and email skipped.', result)
        return
        
    schedule_status = {'processed': 0, 'upserted': 0}
    lineups_status = {'processed_games': 0, 'affected_rows': 0}
    raw_attempted_ids: set[int] = set()
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
            raw_status = RawEvents(config).save_new_finished_games(attempted_ids=raw_attempted_ids)
        if not rebuild_derived:
            refresh_schedule_details(config)
        processed_status = ProcessedEvents(config).save_processed_raw_games()
        key_events_status = KeyEventProcessor(
            config.mongo.url,
            config.mongo.db,
        ).run("backfill", season=config.season.year, stop_on_error=True)
        game_stats_status = GameStats(config).save_game_stats()
        sequence_status = SequenceData(config).save_sequence_data()
    except Exception as exc:
        status = 'FAILED'
        report = f"Pipeline failed: {format_exception(exc)}"
        logger.exception('Pipeline execution failed.')
        try:
            clear_derived_season(config)
        except Exception:
            logger.exception('Failure cleanup also failed for season %s.', config.season.year)
        raise

    with MongoClient(config.mongo.url, serverSelectionTimeoutMS=10000) as client:
        database = client[config.mongo.db]
        schedule = database[config.mongo.collections['collection_schedule']]
        game_ids = schedule.distinct('game_id', {'season': config.season.year})
        projection_counts = publish_app_projections(database, database, config.season.year, game_ids)
        logger.info('Refreshed local app projections: %s', projection_counts)

    atlas_counts = sync_atlas()

    if (
        lineups_status.get('affected_rows', 0) > 0
        or raw_status.get('inserted_rows', 0) > 0
        or processed_status.get('inserted_rows', 0) > 0
        or key_events_status.get('events', 0) > 0
        or game_stats_status.get('team_rows', 0) > 0
        or game_stats_status.get('player_rows', 0) > 0
        or sequence_status.get('shot_rows', 0) > 0
        or sequence_status.get('pass_rows', 0) > 0
        or missing_app_backups(config)
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
    logger.info("Run diagnostics\n%s", report)
    if lineups_status.get('failed_games', 0):
        status = 'PARTIAL'
        logger.warning('Pipeline finished with %s lineup failures.', lineups_status['failed_games'])
    else:
        logger.info('Pipeline finished successfully.')

    with MongoClient(config.mongo.url, serverSelectionTimeoutMS=10000) as index_client:
        ensure_application_indexes(index_client[config.mongo.db])

    if config.email and config.email.password:
        coverage = []
        try:
            with MongoClient(config.mongo.url, serverSelectionTimeoutMS=10000) as client:
                coverage = matchday_coverage(client[config.mongo.db], config, raw_attempted_ids)
        except Exception:
            logger.exception("Could not load matchday coverage for the email report.")

        if any(game.get("game_status") == "finished" and not count
               for _, fixtures in coverage for game, count in fixtures):
            status = "PARTIAL"

        local_counts = {
            config.mongo.collections[key]: stats.get(field, 0)
            for key, stats, field in (
                ("collection_schedule", schedule_status, "upserted"),
                ("collection_lineups", lineups_status, "affected_rows"),
                ("collection_formations", lineups_status, "formation_rows"),
                ("collection_raw_events", raw_status, "inserted_rows"),
                ("collection_processed_events", processed_status, "inserted_rows"),
                ("collection_key_events", key_events_status, "events"),
                ("collection_team_game_stats", game_stats_status, "team_rows"),
                ("collection_player_game_stats", game_stats_status, "player_rows"),
                ("collection_shot_sequences", sequence_status, "shot_rows"),
                ("collection_pass_sequences", sequence_status, "pass_rows"),
            )
        }
        local_counts.update({name: counts.total for name, counts in projection_counts.items()})
        fixture_collections = tuple(local_counts)
        for name in (*config.mongo.collections.values(), "season_config"):
            local_counts.setdefault(name, None)
        body, html_body = format_email(config.season.year, status, coverage, local_counts, atlas_counts)
        if coverage:
            try:
                with MongoClient(config.mongo.url, serverSelectionTimeoutMS=10000) as client:
                    fixture_counts = fixture_collection_counts(
                        client[config.mongo.db], config.season.year, coverage, fixture_collections,
                    )
                details_text, details_html = format_fixture_details(
                    coverage, fixture_collections, fixture_counts,
                )
                body += "\n\n" + details_text
                html_body = html_body.removesuffix("</div>") + details_html + "</div>"
            except Exception:
                logger.exception("Could not build fixture details for the email report.")

        matchdays = ", ".join(matchday for matchday, _ in coverage) or config.season.year
        day = datetime.now().strftime("%d %b %Y")
        subject = f"Bundesliga | {matchdays} | {status} | {day}"
        try:
            send_email(
                smtp_host=config.email.smtp_host,
                smtp_port=config.email.smtp_port,
                username=config.email.username,
                password=config.email.password,
                from_email=config.email.from_email,
                to_email=config.email.to_email,
                subject=subject,
                body=body,
                html_body=html_body,
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
    collection = parser.add_mutually_exclusive_group()
    collection.add_argument(
        '--full', dest='full', action='store_true', default=None,
        help='Collect schedule and raw events, then build stats and sequences (default).',
    )
    collection.add_argument(
        '--incremental', dest='full', action='store_false',
        help='Process existing data without collecting the schedule or raw events.',
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

    parser.add_argument(
        '--atlas-only', action='store_true',
        help='Publish the season already saved in local MongoDB to Atlas without collecting matches.',
    )

    args = parser.parse_args()
    if args.atlas_only and (args.full is True or args.rebuild_derived or args.schedule_only):
        parser.error('--atlas-only cannot be combined with processing or schedule options')
    if args.matchday is not None and not args.schedule_only:
        parser.error('--matchday requires --schedule-only')
    main(season_year=args.season_year, full=args.full is not False, mongo_uri=args.mongo_uri,
         schedule_only=args.schedule_only, matchday=args.matchday,
         rebuild_derived=args.rebuild_derived, atlas_only=args.atlas_only)
