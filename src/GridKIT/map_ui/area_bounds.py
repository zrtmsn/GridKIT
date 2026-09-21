from __future__ import annotations

from math import atan2, cos, radians, sin, sqrt

from pydantic import BaseModel, Field, model_validator


class AreaBounds(BaseModel):
    """
    Bounding box selected in the GUI.

    Coordinate order:
    - south/north are latitudes
    - west/east are longitudes
    """

    south: float = Field(ge=-90, le=90)
    west: float = Field(ge=-180, le=180)
    north: float = Field(ge=-90, le=90)
    east: float = Field(ge=-180, le=180)

    @model_validator(mode="after")
    def validate_box(self) -> "AreaBounds":
        if self.south >= self.north:
            raise ValueError("south must be smaller than north")
        if self.west >= self.east:
            raise ValueError("west must be smaller than east")
        return self

    @property
    def center_lat(self) -> float:
        return (self.south + self.north) / 2

    @property
    def center_lon(self) -> float:
        return (self.west + self.east) / 2

    def contains(self, lat: float, lon: float) -> bool:
        return self.south <= lat <= self.north and self.west <= lon <= self.east

    @property
    def as_overpass_bbox(self) -> str:
        return f"{self.south},{self.west},{self.north},{self.east}"

    @property
    def as_geojson_bbox(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]

    def approx_area_km2(self) -> float:
        height_km = haversine_km(self.south, self.center_lon, self.north, self.center_lon)
        width_km = haversine_km(self.center_lat, self.west, self.center_lat, self.east)
        return height_km * width_km


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0088
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    d_phi = radians(lat2 - lat1)
    d_lambda = radians(lon2 - lon1)

    a = sin(d_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(d_lambda / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return radius_km * c