import unittest

import pandas as pd

from prod_pipeline.sequence_data import SequenceData


def event(idx, event_type, team=1, minute=10, second=0, outcome="Successful", **extra):
    value = {
        "event_idx": idx,
        "type": event_type,
        "team_id": team,
        "team": "Home" if team == 1 else "Away",
        "player_id": idx,
        "player": f"Player {idx}",
        "minute": minute,
        "second": second,
        "period": "FirstHalf",
        "outcome_type": outcome,
        "x": 30.0,
        "y": 50.0,
    }
    value.update(extra)
    return pd.Series(value)


class SequenceOriginTests(unittest.TestCase):
    def classify(self, preceding, first_pass=None):
        if first_pass is None:
            first_pass = event(20, "Pass", minute=12, second=0)
        result = SequenceData._classify_sequence_origin(
            pd.DataFrame([first_pass]), 1, preceding
        )
        return result

    def test_precedence_uses_recovery_after_aerial(self):
        result = self.classify(event(19, "BallRecovery", minute=11, second=9))
        self.assertEqual(result["origin_type"], "Ball recovery")
        self.assertEqual(result["origin_event_id"], 19)

    def test_tackle_recovery_and_interception(self):
        for candidate, expected in [
            (event(19, "Tackle"), "Tackle won"),
            (event(19, "Interception"), "Interception"),
        ]:
            self.assertEqual(self.classify(candidate)["origin_type"], expected)

    def test_restart_pass_is_origin_and_first_pass(self):
        for flag, expected in [
            ("pass_corner", "Corner kick"),
            ("pass_freekick", "Free kick"),
            ("pass_throw_in", "Throw-in"),
            ("pass_goal_kick", "Goal kick"),
            ("pass_kickoff", "First-half kickoff"),
        ]:
            first_pass = event(20, "Pass", minute=12, second=10, **{flag: 1})
            result = self.classify(None, first_pass)
            self.assertEqual(result["origin_type"], expected)
            self.assertEqual(result["origin_event_id"], first_pass["event_idx"])

    def test_restart_flag_from_opponent_cannot_start_this_team_sequence(self):
        first_pass = event(20, "Pass", team=2, minute=12, second=10, pass_throw_in=1)
        result = self.classify(None, first_pass)
        self.assertIsNone(result["origin_type"])

    def test_foul_and_offside_restart(self):
        self.assertEqual(self.classify(event(19, "Foul"))["origin_type"], "Free kick")
        self.assertEqual(self.classify(event(19, "OffsideGiven"))["origin_type"], "Free kick")

    def test_goalkeeper_save_becomes_ball_recovery(self):
        self.assertEqual(self.classify(event(19, "Save"))["origin_type"], "Ball recovery")

    def test_opponent_goal_restart_does_not_borrow_goal_scorer(self):
        result = self.classify(event(19, "Goal", team=2, minute=58, second=56))
        self.assertEqual(result["origin_type"], "Post-goal kickoff")
        self.assertEqual(result["origin_team_id"], 1)
        self.assertIsNone(result["origin_player_id"])
        self.assertIsNone(result["origin_player_name"])
        self.assertIsNone(result["origin_minute"])
        self.assertIsNone(result["origin_second"])

    def test_explicit_post_goal_kickoff_timestamp_is_used(self):
        first_pass = event(30, "Pass", team=1, minute=60, second=40)
        context = pd.DataFrame([
            event(19, "Goal", team=2, minute=58, second=56),
            event(25, "Kickoff", team=1, minute=59, second=48),
            first_pass,
        ])
        result = SequenceData._classify_sequence_origin(
            pd.DataFrame([first_pass]), 1, None, context
        )
        self.assertEqual(result["origin_type"], "Post-goal kickoff")
        self.assertEqual(result["origin_event_id"], 25)
        self.assertEqual(result["origin_team_id"], 1)
        self.assertEqual(result["origin_minute"], 59)
        self.assertEqual(result["origin_second"], 48)

    def test_recorded_kickoff_player_is_kept_for_restarting_team(self):
        kickoff = event(19, "Start", team=1, minute=0, second=0)
        result = self.classify(kickoff)
        self.assertEqual(result["origin_type"], "First-half kickoff")
        self.assertEqual(result["origin_team_id"], 1)
        self.assertEqual(result["origin_player_id"], 19)
        self.assertEqual(result["origin_player_name"], "Player 19")

    def test_opponent_turnover(self):
        result = self.classify(event(19, "Turnover", team=2))
        self.assertEqual(result["origin_type"], "Opponent turnover")

    def test_unsuccessful_opponent_pass_and_touch_are_specific(self):
        for event_type, expected in [
            ("Pass", "Unsuccessful opponent pass"),
            ("BallTouch", "Unsuccessful opponent ball touch"),
        ]:
            result = self.classify(event(19, event_type, team=2, outcome="Unsuccessful"))
            self.assertEqual(result["origin_type"], expected)

    def test_ambiguous_origin_is_unresolved(self):
        result = self.classify(event(19, "Card"))
        self.assertIsNone(result["origin_type"])
        self.assertIn("no possession-establishing", result["origin_evidence"])

    def test_shot_rebound_origin_uses_previous_shot(self):
        previous_shot = event(121, "SavedShot", team=1, minute=5, second=16, shot_event=True)
        goalkeeper_save = event(122, "Save", team=2, minute=5, second=16)
        next_shot = event(123, "SavedShot", team=1, minute=5, second=19, shot_event=True)
        result = SequenceData._classify_sequence_origin(
            pd.DataFrame([next_shot]), 1, None,
            pd.DataFrame([previous_shot, goalkeeper_save, next_shot]),
        )
        self.assertEqual(result["origin_type"], "Shot rebound")
        self.assertEqual(result["origin_event_id"], 121)

    def test_shot_only_without_rebound_evidence_is_unresolved(self):
        shot = event(123, "SavedShot", team=1, shot_event=True)
        result = SequenceData._classify_sequence_origin(
            pd.DataFrame([shot]), 1, None, pd.DataFrame([shot])
        )
        self.assertEqual(result["origin_type"], "Unknown / Unresolved")
        self.assertEqual(result["origin_event_id"], None)

    def test_origin_does_not_change_first_pass_or_duration(self):
        first_pass = event(20, "Pass", minute=12, second=0)
        result = self.classify(event(19, "BallRecovery", minute=11, second=0), first_pass)
        self.assertEqual(result["origin_minute"], 11)
        self.assertEqual(first_pass["minute"], 12)


if __name__ == "__main__":
    unittest.main()
