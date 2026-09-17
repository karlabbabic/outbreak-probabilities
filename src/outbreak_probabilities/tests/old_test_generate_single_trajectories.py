import numpy as np
import pytest

from simulate.generate_single_trajectory import (
    simulate_trajectory,
    calculate_R,
)


def test_default_initial_cases_is_one():
    """
    initial_cases=None should behave as [1]:
    - first day = 1
    - subsequent days can be zero if R=0
    """
    w = [1.0]                # simplest serial interval: all weight at lag 1
    max_weeks = 5
    R = 0.0                  # no new infections after initial

    res = simulate_trajectory(
        w=w,
        max_weeks=max_weeks,
        R=R,
        R_range=None,
        initial_cases=None,  # <- this is what we are testing
        rng=np.random.default_rng(123),
        extinction_window=None,  # avoid early extinction detection
        major_threshold=100,
    )

    traj = res["trajectory"]
    assert traj.shape == (max_weeks,)
    # default initial_cases must seed exactly one case on day 1
    assert traj[0] == 1
    # with R=0, all subsequent days should be zero
    assert np.all(traj[1:] == 0)
    # cumulative should be 1
    assert res["cumulative"] == 1
    # with no extinction_window, and threshold 100, status should be 'ongoing'
    assert res["status"] == "ongoing"
    assert res["PMO"] == 0


def test_custom_initial_cases_used_correctly():
    """
    If initial_cases is provided, it should be used as-is (up to max_weeks),
    and not mutated by the simulator.
    """
    w = [1.0]
    max_weeks = 5
    R = 0.0
    initial = [5, 2, 0]

    res = simulate_trajectory(
        w=w,
        max_weeks=max_weeks,
        R=R,
        R_range=None,
        initial_cases=initial,
        rng=np.random.default_rng(123),
        extinction_window=None,
        major_threshold=100,
    )

    traj = res["trajectory"]
    # first len(initial) days should match
    assert np.array_equal(traj[: len(initial)], np.array(initial))
    # rest should be zeros for R=0
    assert np.all(traj[len(initial) :] == 0)
    # cumulative should be sum(initial)
    assert res["cumulative"] == sum(initial)
    # initial list should not be mutated
    assert initial == [5, 2, 0]


def test_major_threshold_triggered_by_initial_cases():
    """
    If the initial cases alone exceed the major_threshold, we should immediately get status='major' and PMO=1.
    """
    w = [1.0]
    max_weeks = 5
    R = 0.0
    initial_cases = [2]   # cumulative=2

    res = simulate_trajectory(
        w=w,
        max_weeks=max_weeks,
        R=R,
        R_range=None,
        initial_cases=initial_cases,
        rng=np.random.default_rng(123),
        extinction_window=None,
        major_threshold=1,  # smaller than initial cumulative
    )

    assert res["cumulative"] == 2
    assert res["status"] == "major"
    assert res["PMO"] == 1


def test_R_range_when_R_is_none():
    """
    If R is None and R_range is provided, calculate_R should draw from that range.
    Using R_range=(c,c) gives deterministic R=c.
    """
    w = [1.0]
    max_weeks = 3
    R_range = (0.5, 0.5)

    res = simulate_trajectory(
        w=w,
        max_weeks=max_weeks,
        R=None,
        R_range=R_range,
        initial_cases=None,
        rng=np.random.default_rng(123),
        extinction_window=None,
        major_threshold=100,
    )

    assert res["R"] == pytest.approx(0.5)


def test_missing_R_and_R_range_raises():
    """
    If both R and R_range are None, simulation must raise a ValueError.
    """
    w = [1.0]
    with pytest.raises(ValueError):
        simulate_trajectory(
            w=w,
            max_weeks=5,
            R=None,
            R_range=None,
            initial_cases=None,
            rng=np.random.default_rng(123),
            extinction_window=None,
            major_threshold=100,
        )
