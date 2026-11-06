import pandas as pd

from prod_pipeline.sequence_data import SequenceData


def builder() -> SequenceData:
    instance = object.__new__(SequenceData)
    instance.ignore_types = {
        'Save', 'Claim', 'Punch', 'KeeperPickup', 'KeeperSweeper',
        'SubstitutionOn', 'SubstitutionOff', 'Card', 'Smother',
    }
    instance.break_types = {'OffsideGiven', 'OffsideProvoked', 'OffsidePass', 'Foul'}
    instance.opponent_pressure_types = {
        'Aerial', 'BallTouch', 'Challenge', 'Clearance', 'Tackle', 'TakeOn',
    }
    instance.sequence_action_types = {
        'Pass', 'TakeOn', 'BallRecovery', 'Interception', 'Tackle',
        'Aerial', 'BallTouch', 'GoodSkill',
    }
    instance.shot_types = {'Goal', 'SavedShot', 'MissedShots', 'ShotOnPost'}
    instance.base_features = [
        'event_idx', 'type', 'outcome_type', 'team_id', 'team', 'player_id',
        'x', 'y', 'end_x', 'end_y', 'shot_event', 'shot_is_goal',
        'pass_event', 'pass_completed', 'sequence_id', 'sequence_key',
        'sequence_event', 'sequence_team_id', 'sequence_team',
        'sequence_ending', 'sequence_end_event_idx', 'sequence_end_side',
        'sequence_start_reason',
    ]
    instance.shot_feature_keywords = []
    instance.pass_feature_keywords = []
    instance.shot_metadata_features = []
    instance.pass_metadata_features = []
    instance.shared_excluded_features = set()
    instance.shot_excluded_features = set()
    instance.pass_excluded_features = set()
    return instance


def event(event_type, team, outcome='Successful', period='FirstHalf', idx=0):
    values = {
        'game_id': 1,
        'type': event_type,
        'team_id': team,
        'outcome_type': outcome,
        'period': period,
        'event_idx': idx,
        'minute': idx,
        'second': 0,
        'team': f'Team {team}',
        'player_id': idx + 100,
        'player': f'Player {idx}',
        'x': 20.0,
        'y': 30.0,
        'end_x': 25.0,
        'end_y': 35.0,
        'shot_event': event_type in {'Goal', 'SavedShot', 'MissedShots', 'ShotOnPost', 'BlockedShot'},
        'shot_is_goal': event_type == 'Goal',
        'pass_event': event_type == 'Pass',
        'pass_completed': event_type == 'Pass' and outcome == 'Successful',
    }
    return pd.Series(values)


def timeline(*events):
    return pd.DataFrame(events).reset_index(drop=True)


def restart_pass(team, idx, qualifier):
    row = event('Pass', team, idx=idx)
    row[qualifier] = True
    return row


def test_successful_opponent_block_is_a_hard_boundary():
    sequence = timeline(
        event('Pass', 1, idx=1),
        event('BlockedPass', 2, idx=2),
        event('Pass', 1, idx=3),
    )
    subject = builder()
    assert subject._passing_valid_sequence_event(sequence.iloc[1], 1, 'FirstHalf', sequence, 1) == 'break'
    assert subject._shooting_valid_sequence_event(sequence.iloc[1], 1, 'FirstHalf', sequence, 1) == 'break'


def test_successful_opponent_block_followed_by_opponent_control_stops():
    sequence = timeline(
        event('Pass', 1, idx=1),
        event('BlockedPass', 2, idx=2),
        event('Pass', 2, idx=3),
    )
    subject = builder()
    assert subject._passing_valid_sequence_event(sequence.iloc[1], 1, 'FirstHalf', sequence, 1) == 'break'
    assert subject._shooting_valid_sequence_event(sequence.iloc[1], 1, 'FirstHalf', sequence, 1) == 'break'


def test_unsuccessful_opponent_tackle_does_not_transfer_possession_alone():
    sequence = timeline(
        event('Pass', 1, idx=1),
        event('Tackle', 2, 'Unsuccessful', idx=2),
        event('BallTouch', 1, 'Successful', idx=3),
    )
    subject = builder()
    assert subject._passing_valid_sequence_event(sequence.iloc[1], 1, 'FirstHalf', sequence, 1) == 'ignore'


def test_successful_aerial_without_control_evidence_is_conservative_boundary():
    sequence = timeline(
        event('Pass', 1, idx=1),
        event('Aerial', 2, 'Successful', idx=2),
    )
    subject = builder()
    assert subject._passing_valid_sequence_event(sequence.iloc[1], 1, 'FirstHalf', sequence, 1) == 'break'


def test_opponent_error_before_same_team_shot_continues_backward_scan():
    sequence = timeline(
        event('Pass', 1, idx=1),
        event('Error', 2, 'Successful', idx=2),
        event('SavedShot', 1, 'Successful', idx=3),
    )
    subject = builder()
    assert subject._shooting_valid_sequence_event(sequence.iloc[1], 1, 'FirstHalf', sequence, 1) == 'ignore'


def test_opponent_error_is_not_unconditionally_ignored():
    sequence = timeline(
        event('Pass', 1, idx=1),
        event('Error', 2, 'Successful', idx=2),
        event('Pass', 2, 'Successful', idx=3),
    )
    subject = builder()
    assert subject._shooting_valid_sequence_event(sequence.iloc[1], 1, 'FirstHalf', sequence, 1) == 'break'


def test_unsuccessful_opponent_aerial_before_same_team_shot_continues_backward_scan():
    sequence = timeline(
        event('Pass', 1, idx=1),
        event('Aerial', 2, 'Unsuccessful', idx=2),
        event('SavedShot', 1, 'Successful', idx=3),
    )
    subject = builder()
    assert subject._shooting_valid_sequence_event(sequence.iloc[1], 1, 'FirstHalf', sequence, 1) == 'ignore'


def test_unsuccessful_same_team_pass_followed_by_recovery_continues():
    sequence = timeline(
        event('Pass', 1, 'Unsuccessful', idx=1),
        event('BallRecovery', 1, 'Successful', idx=2),
        event('Pass', 1, 'Successful', idx=3),
    )
    subject = builder()
    assert subject._passing_valid_sequence_event(sequence.iloc[0], 1, 'FirstHalf', sequence, 0) == 'keep'
    assert subject._shooting_valid_sequence_event(sequence.iloc[0], 1, 'FirstHalf', sequence, 0) == 'keep'


def test_stoppage_and_period_change_break_continuity():
    subject = builder()
    foul = event('Foul', 1, idx=2)
    assert subject._passing_valid_sequence_event(foul, 1, 'FirstHalf') == 'break'
    period_change = event('Pass', 1, idx=3, period='SecondHalf')
    assert subject._passing_valid_sequence_event(period_change, 1, 'FirstHalf') == 'break'


def test_opponent_restart_cannot_be_sequence_origin():
    sequence = timeline(event('Pass', 1, idx=20))
    context = timeline(
        event('Pass', 2, 'Unsuccessful', idx=19),
        event('Pass', 1, 'Successful', idx=20),
    )
    context.loc[0, 'pass_throw_in'] = True
    metadata = SequenceData._classify_sequence_origin(sequence, 1, context.iloc[0], context)
    assert metadata['origin_type'] != 'Throw-in'


def test_passing_builder_excludes_shot_ending_chain_but_keeps_non_shot_chain():
    subject = builder()
    events = pd.DataFrame([
        event('Pass', 1, idx=1),
        event('BallTouch', 1, idx=2),
        event('Pass', 1, idx=3),
        event('SavedShot', 1, idx=4),
        event('Pass', 2, idx=5),
        event('Pass', 2, idx=6),
        event('BallRecovery', 1, idx=7),
    ])
    result = subject.build_passing_sequences_for_game(events)
    assert not result.empty
    assert set(result['sequence_ending']) == {'BallRecovery'}
    assert result.groupby('sequence_id').size().tolist() == [2]
    assert not result['sequence_ending'].isin({'Shot', 'Goal'}).any()


def test_shot_builder_keeps_direct_shots_and_represents_each_once():
    subject = builder()
    events = pd.DataFrame([
        event('SavedShot', 1, idx=1),
        event('Pass', 2, idx=2),
        event('Goal', 2, idx=3),
    ])
    result = subject.build_shot_sequences_for_game(events)
    assert result['sequence_id'].nunique() == 2
    assert set(result.groupby('sequence_id')['sequence_start_reason'].first()) == {
        'Shot without prior successful pass',
        'First qualifying pass',
    }


def test_restart_pass_splits_same_team_passing_phases():
    subject = builder()
    events = timeline(
        event('Pass', 1, idx=1),
        event('Pass', 1, idx=2),
        restart_pass(1, 3, 'pass_corner'),
        event('Pass', 1, idx=4),
        event('Pass', 1, idx=5),
        event('OffsidePass', 1, idx=6),
    )
    result = subject.build_passing_sequences_for_game(events)
    assert result.groupby('sequence_id')['event_idx'].apply(list).tolist() == [[1, 2], [3, 4, 5]]
    assert result.groupby('sequence_id')['sequence_ending'].first().tolist() == ['Corner kick', 'Offside']
    assert SequenceData._restart_type(events.iloc[2], 1) == 'Corner kick'


def test_restart_qualifiers_are_hard_boundaries_for_all_restart_types():
    subject = builder()
    for qualifier, expected in [
        ('pass_corner', 'Corner kick'),
        ('pass_throw_in', 'Throw-in'),
        ('pass_freekick', 'Free kick'),
        ('pass_goal_kick', 'Goal kick'),
        ('pass_kickoff', 'Kickoff'),
        ('pass_penalty', 'Penalty kick'),
    ]:
        row = restart_pass(1, 10, qualifier)
        assert subject._restart_type(row, 1) == expected


def test_shot_buildup_stops_backward_at_restart_but_keeps_restart_pass():
    subject = builder()
    events = timeline(
        event('Pass', 1, idx=1),
        event('Pass', 1, idx=2),
        restart_pass(1, 3, 'pass_corner'),
        event('Pass', 1, idx=4),
        event('SavedShot', 1, idx=5),
    )
    result = subject.build_shot_sequences_for_game(events)
    assert result['event_idx'].tolist() == [3, 4, 5]
    assert SequenceData._restart_type(events.iloc[2], 1) == 'Corner kick'


def test_every_shot_is_a_hard_boundary_including_blocked_shots():
    subject = builder()
    events = timeline(
        event('Pass', 1, idx=1),
        event('Pass', 1, idx=2),
        event('BlockedShot', 1, idx=3),
        event('BallRecovery', 1, idx=4),
        event('Pass', 1, idx=5),
        event('Goal', 1, idx=6),
    )

    shot_result = subject.build_shot_sequences_for_game(events)
    groups = shot_result.groupby('sequence_id')
    assert groups['event_idx'].apply(list).tolist() == [[1, 2, 3], [4, 5, 6]]

    pass_result = subject.build_passing_sequences_for_game(events)
    assert pass_result.empty


def test_post_shot_non_shot_chain_is_independent_from_pre_shot_chain():
    subject = builder()
    events = timeline(
        event('Pass', 1, idx=1),
        event('Pass', 1, idx=2),
        event('SavedShot', 1, idx=3),
        event('BallRecovery', 1, idx=4),
        event('Pass', 1, idx=5),
        event('Pass', 1, idx=6),
        event('BallTouch', 1, idx=7),
    )

    result = subject.build_passing_sequences_for_game(events)
    assert result.groupby('sequence_id')['event_idx'].apply(list).tolist() == [[5, 6]]
    assert not result['sequence_ending'].isin({'Shot', 'Goal'}).any()
