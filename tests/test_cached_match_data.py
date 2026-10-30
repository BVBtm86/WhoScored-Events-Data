import unittest
from types import SimpleNamespace

import pandas as pd

from prod_pipeline.game_lineups import GameLineups
from prod_pipeline.game_schedule import GameSchedule, match_details


class CachedMatchDataTests(unittest.TestCase):
    def test_minutes_include_substitutions_dismissals_and_unused_bench(self):
        data = {
            'expandedMaxMinute': 96,
            'home': {
                'teamId': 1,
                'incidentEvents': [{'playerId': 1, 'expandedMinute': 50,
                                    'cardType': {'displayName': 'Red'}}],
                'players': [
                    {'playerId': 1, 'isFirstEleven': True},
                    {'playerId': 2, 'subbedInExpandedMinute': 60, 'subbedOutExpandedMinute': 80},
                    {'playerId': 3},
                    {'playerId': 4, 'isFirstEleven': True},
                    {'playerId': 5, 'subbedInExpandedMinute': 90},
                ],
            },
            'away': {'teamId': 2, 'players': []},
        }
        rows = GameLineups._players_from_match(data)
        self.assertEqual(rows.minutes_played.tolist(), [50, 20, 0, 96, 6])
        self.assertEqual(rows.is_starter.tolist(), [True, False, False, True, False])

    def test_match_metadata_uses_both_managers_and_actual_event_presence(self):
        details = match_details(10, {
            'home': {'managerName': 'Home manager'}, 'away': {'managerName': 'Away manager'},
            'events': [{'id': 1}], 'venueName': 'Arena', 'attendance': 0,
            'referee': {'name': 'Referee'},
        })
        self.assertEqual(details['away_manager'], 'Away manager')
        self.assertEqual(details['venue'], 'Arena')
        self.assertEqual(details['attendance'], 0)
        self.assertTrue(details['stats_available'])
        self.assertFalse(match_details(10, {'events': []})['stats_available'])

    def test_scheduled_match_has_no_result_or_source_status(self):
        runner = GameSchedule.__new__(GameSchedule)
        runner.config = SimpleNamespace(
            season=SimpleNamespace(year='2026-2027', name='Bundesliga', country='Germany'),
            schedule_games=SimpleNamespace(finished_status_code=6),
        )
        doc = runner._build_docs(pd.DataFrame([{
            'game_id': 10, 'week': 8, 'status': 1,
            'home_score': 0, 'away_score': 0, 'stats_available': False,
        }]))[0]
        self.assertEqual(doc['game_status'], 'scheduled')
        self.assertIsNone(doc['home_goals'])
        self.assertIsNone(doc['away_goals'])
        self.assertNotIn('source_status', doc)


if __name__ == '__main__':
    unittest.main()
