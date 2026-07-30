# dashboard/test_app.py
import core.constants as const
from GridKIT.dashboard.app import _LOAD_COLORS, _load_style


def test_overloaded_cable_is_red_and_thickest():
    color, weight = _load_style(1.05)
    assert color == "#d7191c"
    assert weight == max(w for _, _, w in _LOAD_COLORS)


def test_severity_bands_are_distinct():
    bands = [_load_style(pu)[0] for pu in (1.20, 0.95, 0.80, 0.20)]
    assert len(set(bands)) == 4, "each severity band needs its own colour"


def test_weight_increases_with_loading():
    weights = [_load_style(pu)[1] for pu in (0.2, 0.8, 0.95, 1.2)]
    assert weights == sorted(weights)


def test_exactly_at_the_overload_threshold_is_not_yet_red():
    # the environment curtails on `> threshold`, so the map must agree — a cable
    # sitting exactly at rating has not tripped
    color, _ = _load_style(const.LINE_OVERLOAD_THRESHOLD)
    assert color != "#d7191c"


def test_watch_level_still_renders_a_colour():
    # anything recorded at all is drawn coloured; grey is reserved for lines the
    # runner never recorded (below the watch level)
    color, _ = _load_style(const.LINE_WATCH_THRESHOLD)
    assert color.startswith("#")


def test_a_cable_that_tripped_reads_as_overloaded_even_when_its_mean_is_lower():
    # peaks are averaged over seeds, so a cable that violated its rating in a minority
    # of seeds sits below 1.0 on average — the map must not contradict the trip count
    plain, _ = _load_style(0.85)
    tripped, weight = _load_style(0.85, tripped=True)
    assert plain != tripped
    assert tripped == "#d7191c"
    assert weight == max(w for _, _, w in _LOAD_COLORS)
