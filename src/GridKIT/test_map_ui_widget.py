from map_ui.map_widget import bounds_from_drawings


def test_bounds_from_drawings_polygon():
    map_data = {
        "last_active_drawing": {
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [8.3900, 49.0000],
                        [8.4100, 49.0000],
                        [8.4100, 49.0100],
                        [8.3900, 49.0100],
                        [8.3900, 49.0000],
                    ]
                ],
            }
        }
    }

    bounds = bounds_from_drawings(map_data)

    assert bounds is not None
    assert bounds.south == 49.0000
    assert bounds.west == 8.3900
    assert bounds.north == 49.0100
    assert bounds.east == 8.4100


def test_bounds_from_drawings_empty_data():
    bounds = bounds_from_drawings(None)

    assert bounds is None


def test_bounds_from_drawings_without_drawing():
    map_data = {
        "last_active_drawing": None,
        "all_drawings": [],
    }

    bounds = bounds_from_drawings(map_data)

    assert bounds is None