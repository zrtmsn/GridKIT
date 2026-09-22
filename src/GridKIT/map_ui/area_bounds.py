from __future__ import annotations

from math import atan2, cos, radians, sin, sqrt

from pydantic import BaseModel, Field, model_validator


MIN_LATITUDE = -90.0
MAX_LATITUDE = 90.0
MIN_LONGITUDE = -180.0
MAX_LONGITUDE = 180.0

EARTH_RADIUS_KM = 6371.0088
AVERAGE_DIVISOR = 2.0
HAVERSINE_ANGLE_DIVISOR = 2.0
HAVERSINE_CENTRAL_ANGLE_FACTOR = 2.0


class AreaBounds(BaseModel):
    """
    Bounding box selected in the map UI.

    Coordinate order:
    - south/north are latitudes
    - west/east are longitudes
    """

    south: float = Field(ge=MIN_LATITUDE, le=MAX_LATITUDE)
    west: float = Field(ge=MIN_LONGITUDE, le=MAX_LONGITUDE)
    north: float = Field(ge=MIN_LATITUDE, le=MAX_LATITUDE)
    east: float = Field(ge=MIN_LONGITUDE, le=MAX_LONGITUDE)

    @model_validator(mode="after")
    def validate_box(self) -> "AreaBounds":
        """Ensure that the bounding box has a positive area."""
        if self.south >= self.north:
            raise ValueError("south must be smaller than north")
        if self.west >= self.east:
            raise ValueError("west must be smaller than east")
        return self

    @property
    def center_lat(self) -> float:
        """Return the latitude of the bounding box center."""
        return (self.south + self.north) / AVERAGE_DIVISOR

    @property
    def center_lon(self) -> float:
        """Return the longitude of the bounding box center."""
        return (self.west + self.east) / AVERAGE_DIVISOR

    def contains(self, lat: float, lon: float) -> bool:
        """Check whether a coordinate lies inside the bounding box."""
        return self.south <= lat <= self.north and self.west <= lon <= self.east

    @property
    def as_overpass_bbox(self) -> str:
        """Return the bounding box in Overpass API order."""
        return f"{self.south},{self.west},{self.north},{self.east}"

    @property
    def as_geojson_bbox(self) -> list[float]:
        """Return the bounding box in GeoJSON order."""
        return [self.west, self.south, self.east, self.north]

    def approx_area_km2(self) -> float:
        """Approximate the bounding box area in square kilometers."""
        height_km = haversine_km(self.south, self.center_lon, self.north, self.center_lon)
        width_km = haversine_km(self.center_lat, self.west, self.center_lat, self.east)
        return height_km * width_km


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two coordinates."""
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    d_phi = radians(lat2 - lat1)
    d_lambda = radians(lon2 - lon1)

    a = (
        sin(d_phi / HAVERSINE_ANGLE_DIVISOR) ** 2
        + cos(phi1)
        * cos(phi2)
        * sin(d_lambda / HAVERSINE_ANGLE_DIVISOR) ** 2
    )
    c = HAVERSINE_CENTRAL_ANGLE_FACTOR * atan2(sqrt(a), sqrt(1 - a))

    return EARTH_RADIUS_KM * c