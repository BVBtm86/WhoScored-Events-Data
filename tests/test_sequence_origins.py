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
            ("pass_kickoff", "Kickoff"),
        ]:
            first_pass = event(20, "Pass", minute=12, second=10, **{flag: 1})
            result = self.classify(None, first_pass)
            self.assertEqual(result["origin_type"], expected)
            self.assertEqual(result["origin_event_id"], first_pass["event_idx"])

    def test_foul_and_offside_restart(self):
        self.assertEqual(self.classify(event(19, "Foul"))["origin_type"], "Free kick")
        self.assertEqual(self.classify(event(19, "OffsideGiven"))["origin_type"], "Free kick")

    def test_goalkeeper_save_becomes_ball_recovery(self):
        self.assertEqual(self.classify(event(19, "Save"))["origin_type"], "Ball recovery")

    def test_opponent_turnover(self):
        result = self.classify(event(19, "Turnover", team=2))
        self.assertEqual(result["origin_type"], "Opponent turnover")

    def test_ambiguous_origin_is_unresolved(self):
        result = self.classify(event(19, "Card"))
        self.assertIsNone(result["origin_type"])
        self.assertIn("no possession-establishing", result["origin_evidence"])

    def test_origin_does_not_change_first_pass_or_duration(self):
        first_pass = event(20, "Pass", minute=12, second=0)
        result = self.classify(event(19, "BallRecovery", minute=11, second=0), first_pass)
        self.assertEqual(result["origin_minute"], 11)
        self.assertEqual(first_pass["minute"], 12)


if __name__ == "__main__":
    unittest.main()
