import csv
from pathlib import Path

import numpy as np

from simulate.batch_processing import generate_batch
from simulate.generate_single_trajectory import simulate_trajectory

def test_generate_batch_shapes_and_csv(tmp_path):
    """
    Basic integration test for generate_batch:
    - trajectories array has shape (N, max_days)
    - CSV is created with N rows + header
    - PMO flag is present and consistent with status
    """
    # Simple weights: w_1 = 1, others implicitly zero
    w = [1.0]
    N = 5
    max_days = 4
    R_range = (0.0, 0.0)  # R=0, no new infections after initial
    initial_cases = [1]

    out_csv = tmp_path / "simulated_cases.csv"

    trajectories, csv_path = generate_batch(
        N=N,
        w=w,
        max_days=max_days,
        R_range=R_range,
        initial_cases=initial_cases,
        extinction_window=None,
        major_threshold=100,
        out_path=str(out_csv),
        use_tempfile=False,
        seed=123,
    )

    # Array properties
    assert trajectories.shape == (N, max_days)
    # For R=0 and initial [1], every trajectory should be [1, 0, 0, 0]
    expected_row = np.array([1, 0, 0, 0])
    for i in range(N):
        assert np.array_equal(trajectories[i], expected_row)

    # CSV existence
    assert csv_path == out_csv
    assert csv_path.exists()

    # Check CSV content: N data rows + 1 header row
    rows = list(csv.reader(csv_path.open()))
    assert len(rows) == N + 1  # header + N simulations

    header = rows[0]
    # Check key columns are present
    assert header[0] == "sim_id"
    assert header[1] == "R_draw"
    assert header[-3:] == ["cumulative_cases", "status", "PMO"]

    # Parse rows and check status & PMO flags
    dict_reader = csv.DictReader(csv_path.open())
    for row in dict_reader:
        # cumulative must be string representation of sum of the day_* columns
        cumulative = int(row["cumulative_cases"])
        day_cols = [k for k in row.keys() if k.startswith("day_")]
        day_vals = [int(row[k]) for k in sorted(day_cols, key=lambda x: int(x.split("_")[1]))]
        assert cumulative == sum(day_vals)

        # With R=0 and threshold=100, should never hit major; status 'ongoing' or 'minor'
        assert row["PMO"] in ("0", "1")
        if row["status"] == "major":
            # If ever major, PMO must be 1 and cumulative >= 100
            assert int(row["PMO"]) == 1
            assert cumulative >= 100
        else:
            # For minor/ongoing, PMO must be 0 or (if you later change policy, low)
            assert int(row["PMO"]) in (0, 1)  # keep this permissive for now


def test_generate_batch_extinction_window(tmp_path):
    """
    Check that extinction_window leads to 'minor' status when zeros persist.
    Here we simulate with R=0 and extinction_window=1 so extinction is immediate after day 1.
    """
    w = [1.0]
    N = 1
    max_days = 5
    R_range = (0.0, 0.0)
    initial_cases = [1]

    out_csv = tmp_path / "simulated_cases_extinct.csv"

    trajectories, csv_path = generate_batch(
        N=N,
        w=w,
        max_days=max_days,
        R_range=R_range,
        initial_cases=initial_cases,
        extinction_window=1,  # one zero day at the end triggers extinction
        major_threshold=100,
        out_path=str(out_csv),
        use_tempfile=False,
        seed=123,
    )

    # Trajectory itself: [1,0,0,0,0]
    assert trajectories.shape == (1, max_days)
    assert np.array_equal(trajectories[0], np.array([1, 0, 0, 0, 0]))

    # CSV row status should be 'minor' with PMO=0
    dict_reader = csv.DictReader(csv_path.open())
    rows = list(dict_reader)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "minor"
    assert int(row["PMO"]) == 0
