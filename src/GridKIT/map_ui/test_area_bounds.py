import pytest
from pydantic import ValidationError

from map_ui.area_bounds import AreaBounds


def test_area_bounds_accepts_valid_bbox():
    bounds = AreaBounds(
        south=49.0,
        west=8.4,
        north=49.01,
        east=8.41,
    )

    assert bounds.south == 49.0
    assert bounds.west == 8.4
    assert bounds.north == 49.01
    assert bounds.east == 8.41


def test_area_bounds_rejects_invalid_latitude_order():
    with pytest.raises(ValidationError):
        AreaBounds(
            south=49.01,
            west=8.4,
            north=49.0,
            east=8.41,
        )


def test_area_bounds_rejects_invalid_longitude_order():
    with pytest.raises(ValidationError):
        AreaBounds(
            south=49.0,
            west=8.41,
            north=49.01,
            east=8.4,
        )


def test_area_bounds_contains_point_inside_and_outside():
    bounds = AreaBounds(
        south=49.0,
        west=8.4,
        north=49.01,
        east=8.41,
    )

    assert bounds.contains(49.005, 8.405) is True
    assert bounds.contains(48.999, 8.405) is False
    assert bounds.contains(49.005, 8.411) is False


def test_area_bounds_overpass_bbox_order_is_south_west_north_east():
    bounds = AreaBounds(
        south=49.0,
        west=8.4,
        north=49.01,
        east=8.41,
    )

    assert bounds.as_overpass_bbox == "49.0,8.4,49.01,8.41"


def test_area_bounds_geojson_bbox_order_is_west_south_east_north():
    bounds = AreaBounds(
        south=49.0,
        west=8.4,
        north=49.01,
        east=8.41,
    )

    assert bounds.as_geojson_bbox == [8.4, 49.0, 8.41, 49.01]


def test_area_bounds_approx_area_is_positive():
    bounds = AreaBounds(
        south=49.0,
        west=8.4,
        north=49.01,
        east=8.41,
    )

    assert bounds.approx_area_km2() > 0