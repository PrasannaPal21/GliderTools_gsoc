import numpy as np

from glidertools.calibration import bottle_matchup


def test_bottle_matchup_match():
    # one dive, 5 depth points, bottle sample close in time and depth
    gld_dives = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    gld_depth = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    base = np.datetime64("2020-01-01T12:00")
    gld_time = np.array([base + np.timedelta64(i, "m") for i in range(5)])

    # bottle at ~30m, taken 10 min after glider start
    btl_depth = np.array([30.0])
    btl_time = np.array([base + np.timedelta64(10, "m")])
    btl_values = np.array([99.9])

    result = bottle_matchup(
        gld_dives, gld_depth, gld_time,
        btl_depth, btl_time, btl_values,
    )

    # should match at index 2 (depth=30)
    assert result[2] == 99.9
    # everything else should be nan
    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert np.isnan(result[3])
    assert np.isnan(result[4])


def test_bottle_matchup_no_match_time():
    # bottle sample too far in time (>120 min default)
    gld_dives = np.array([1.0, 1.0, 1.0])
    gld_depth = np.array([10.0, 20.0, 30.0])
    base = np.datetime64("2020-01-01T12:00")
    gld_time = np.array([base + np.timedelta64(i, "m") for i in range(3)])

    btl_depth = np.array([20.0])
    btl_time = np.array([base + np.timedelta64(200, "m")])  # 200 min away
    btl_values = np.array([50.0])

    result = bottle_matchup(
        gld_dives, gld_depth, gld_time,
        btl_depth, btl_time, btl_values,
    )
    # nothing should match
    assert np.all(np.isnan(result))


def test_bottle_matchup_no_match_depth():
    # bottle close in time but depth diff > 5m threshold
    gld_dives = np.array([1.0, 1.0, 1.0])
    gld_depth = np.array([10.0, 20.0, 30.0])
    base = np.datetime64("2020-01-01T12:00")
    gld_time = np.array([base + np.timedelta64(i, "m") for i in range(3)])

    btl_depth = np.array([100.0])  # way deeper than any glider point
    btl_time = np.array([base + np.timedelta64(1, "m")])
    btl_values = np.array([50.0])

    result = bottle_matchup(
        gld_dives, gld_depth, gld_time,
        btl_depth, btl_time, btl_values,
    )
    assert np.all(np.isnan(result))
