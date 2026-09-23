import pytest
from pydantic import ValidationError

from map_ui.area_bounds import AreaBounds


# Shared test fixture for a small valid bounding box near Karlsruhe.
VALID_SOUTH = 49.0
VALID_WEST = 8.4
VALID_NORTH = 49.01
VALID_EAST = 8.41

POINT_INSIDE_LAT = 49.005
POINT_INSIDE_LON = 8.405
POINT_OUTSIDE_SOUTH_LAT = 48.999
POINT_OUTSIDE_EAST_LON = 8.411

EXPECTED_OVERPASS_BBOX = "49.0,8.4,49.01,8.41"
EXPECTED_GEOJSON_BBOX = [VALID_WEST, VALID_SOUTH, VALID_EAST, VALID_NORTH]

MIN_POSITIVE_AREA_KM2 = 0.0


def make_valid_bounds() -> AreaBounds:
    """Create the valid bounding box used by the tests."""
    return AreaBounds(
        south=VALID_SOUTH,
        west=VALID_WEST,
        north=VALID_NORTH,
        east=VALID_EAST,
    )


def test_area_bounds_accepts_valid_bbox():
    bounds = make_valid_bounds()

    assert bounds.south == VALID_SOUTH
    assert bounds.west == VALID_WEST
    assert bounds.north == VALID_NORTH
    assert bounds.east == VALID_EAST


def test_area_bounds_rejects_invalid_latitude_order():
    with pytest.raises(ValidationError):
        AreaBounds(
            south=VALID_NORTH,
            west=VALID_WEST,
            north=VALID_SOUTH,
            east=VALID_EAST,
        )


def test_area_bounds_rejects_invalid_longitude_order():
    with pytest.raises(ValidationError):
        AreaBounds(
            south=VALID_SOUTH,
            west=VALID_EAST,
            north=VALID_NORTH,
            east=VALID_WEST,
        )


def test_area_bounds_contains_point_inside_and_outside():
    bounds = make_valid_bounds()

    assert bounds.contains(POINT_INSIDE_LAT, POINT_INSIDE_LON) is True
    assert bounds.contains(POINT_OUTSIDE_SOUTH_LAT, POINT_INSIDE_LON) is False
    assert bounds.contains(POINT_INSIDE_LAT, POINT_OUTSIDE_EAST_LON) is False


def test_area_bounds_overpass_bbox_order_is_south_west_north_east():
    bounds = make_valid_bounds()

    assert bounds.as_overpass_bbox == EXPECTED_OVERPASS_BBOX


def test_area_bounds_geojson_bbox_order_is_west_south_east_north():
    bounds = make_valid_bounds()

    assert bounds.as_geojson_bbox == EXPECTED_GEOJSON_BBOX


def test_area_bounds_approx_area_is_positive():
    bounds = make_valid_bounds()

    assert bounds.approx_area_km2() > MIN_POSITIVE_AREA_KM2