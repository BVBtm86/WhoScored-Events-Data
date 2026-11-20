"""Compact MongoDB documents consumed by the Bundesliga app."""

from typing import Any


SOURCE_EVENTS = "game_processed_events"
SOURCE_SHOTS = "game_shot_sequences"
SOURCE_PASSES = "game_pass_sequences"
SOURCE_TEAM_STATS = "game_team_stats"
SOURCE_PLAYER_STATS = "game_player_stats"

TARGET_EVENTS = "game_processed_events_app"
TARGET_SHOTS = "game_shot_sequences_app"
TARGET_PASSES = "game_pass_sequences_app"
TARGET_TEAM_STATS = "game_team_stats_app"
TARGET_PLAYER_STATS = "game_player_stats_app"

EVENT_FIELDS = (
    "game_id", "season", "period", "minute", "second", "expanded_minute", "event_idx",
    "team_id", "team", "player_id", "player", "recipient_id", "pass_recipient_id",
    "related_player_id", "type", "event_type", "outcome_type", "x", "y", "end_x", "end_y",
    "goal_mouth_y", "goal_mouth_z", "blocked_x", "blocked_y", "shot_result",
    "pass_completed", "pass_attempt", "pass_successful", "shot_goal", "shot_on_target",
    "shot_blocked", "shot_own_goal", "shot_woodwork", "progressive_pass",
)

EVENT_CATEGORY_FLAGS = (
    ("pass", "pass_event"),
    ("shot", "shot_event"),
    ("dribble", "dribble_event"),
    ("tackle", "tackle_attempted_event"),
    ("interception", "interception_event"),
    ("clearance", "clearance_event"),
    ("block", "block_event"),
    ("offside", "offside_event"),
    ("foul", "foul_event"),
    ("aerial", "aerial_duel_event"),
    ("touch", "touch_event"),
    ("loss", "loss_possession_event"),
    ("recovery", "ball_recovery_event"),
    ("error", "error_event"),
    ("save", "save_event"),
    ("claim", "claim_event"),
    ("punch", "punch_event"),
    ("keeper_pickup", "keeper_pickup"),
    ("keeper_sweeper", "keeper_sweeper"),
    ("card", "card_event"),
    ("substitution", "substitution_event"),
)

SEQUENCE_ACTION_FIELDS = (
    "event_idx", "sequence_event", "period", "minute", "second",
    "team_id", "team", "player_id", "player", "type", "event_type", "outcome_type",
    "x", "y", "end_x", "end_y", "pass_completed", "shot_goal", "shot_own_goal",
    "shot_result",
)

SEQUENCE_META_FIELDS = (
    "game_id", "season", "sequence_id", "sequence_key", "sequence_ending",
    "sequence_team_id", "sequence_team", "origin_type", "origin_event_id", "origin_event_type", "origin_player_id",
    "origin_player_name", "origin_team_id", "origin_minute", "origin_second", "origin_x",
    "origin_y", "origin_evidence",
)

TEAM_STAT_KEYS = (
    "goals", "own_goals", "shots", "shot_on_target", "shot_off_target", "shot_blocked", "shot_woodwork",
    "shot_zone_inside_box", "shot_zone_outside_box", "shot_set_piece", "shot_fastbreak", "shot_open_play",
    "shot_penalty", "possession", "touches", "loss_of_possession", "dispossessed", "turnover", "pass_attempt",
    "pass_completed", "pass_completion_pct", "pass_key_pass", "pass_assist", "pass_through_ball", "pass_forward",
    "pass_backward", "pass_cross", "pass_cross_completed", "pass_cross_completion_pct", "pass_corner", "pass_long",
    "pass_long_completed", "pass_long_completion_pct", "pass_final_third", "pass_final_third_completed",
    "pass_final_third_completion_pct", "tackles_attempted", "tackle_gained_possession", "tackle_success_pct",
    "interceptions", "clearances", "blocks", "blocked_shot", "errors", "ball_recoveries", "dribbles",
    "dribble_successful", "offsides", "foul_suffered", "foul_committed", "aerial_duels", "aerial_duel_won",
    "saves", "goalkeeper_actions", "claims", "punches", "keeper_pickup", "keeper_sweeper", "cards",
    "yellow_card", "second_yellow_card", "red_card", "shot_accuracy", "goal_conversion_pct",
)
PLAYER_STAT_KEYS = (
    "shot_goal", "shots", "shot_on_target", "pass_key_pass", "pass_assist", "dribble_successful",
    "pass_completed", "pass_attempt", "pass_completion_pct", "pass_long", "pass_long_completed", "pass_final_third",
    "pass_cross", "pass_cross_completed", "tackles_attempted", "tackle_gained_possession", "interceptions",
    "clearances", "blocks", "aerial_duel_won", "aerial_duels", "aerial_duel_win_pct", "touches",
    "ball_recoveries", "loss_of_possession", "foul_committed", "foul_suffered", "yellow_card", "red_card",
    "offsides", "saves", "claims", "punches", "keeper_sweeper", "keeper_pickup", "goalkeeper_actions",
)
PLAYER_PERIODS = ("ft", "fh", "sh")


def compact(document: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: document[field] for field in fields if field in document}


def app_event(document: dict[str, Any]) -> dict[str, Any]:
    projected = compact(document, EVENT_FIELDS)
    projected["categories"] = [
        category for category, source_flag in EVENT_CATEGORY_FLAGS
        if document.get(source_flag) is True
    ]
    return projected


def compact_stats(stats: Any, periods: tuple[str, ...], keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    if not isinstance(stats, dict):
        return {}
    return {
        period: {key: stats[period][key] for key in keys if key in stats[period]}
        for period in periods
        if isinstance(stats.get(period), dict)
    }


def app_team_stats(document: dict[str, Any]) -> dict[str, Any]:
    projected = compact(document, ("game_id", "team_id"))
    projected["stats"] = compact_stats(document.get("stats"), ("ft", "fh", "sh", "m_1_15", "m_16_30", "m_31_45", "m_46_60", "m_61_75", "m_76_90", "m_1_30", "m_31_60", "m_61_90"), TEAM_STAT_KEYS)
    return projected


def app_player_stats(document: dict[str, Any]) -> dict[str, Any]:
    fields = ("game_id", "team_id", "player_id", "player", "player_name", "position", "starting_lineup", "minutes_played")
    projected = compact(document, fields)
    projected["stats"] = compact_stats(document.get("stats"), PLAYER_PERIODS, PLAYER_STAT_KEYS)
    return projected


def sequence_documents(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("sequence_key")
        if not key:
            continue
        sequence = grouped.setdefault(key, compact(row, SEQUENCE_META_FIELDS))
        sequence.setdefault("actions", []).append(compact(row, SEQUENCE_ACTION_FIELDS))
    for sequence in grouped.values():
        sequence["actions"].sort(key=lambda action: (
            action.get("event_idx", 0), action.get("sequence_event", 0)
        ))
        sequence["_id"] = sequence["sequence_key"]
        sequence["action_count"] = len(sequence["actions"])
        sequence["completed_pass_count"] = sum(
            bool(action.get("pass_event")) and bool(action.get("pass_completed"))
            for action in sequence["actions"]
        )
    return list(grouped.values())
