"""Offline elevation raster loading, sampling, and terrain statistics."""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians
from pathlib import Path

import numpy as np


@dataclass
class ElevationGrid:
    elevations: np.ndarray
    south: float
    west: float
    arc_seconds: float
    crs: str = "EPSG:4326"
    geo_transform: tuple[float, float, float, float, float, float] | None = None
    raster_bounds: dict[str, float] | None = None
    pixel_size_m: tuple[float, float] | None = None
    vertical_units: str = "m"
    source: str = "srtm"
    nodata: float | None = None

    @property
    def north(self) -> float:
        if self.raster_bounds is not None:
            return self.raster_bounds["north"]
        return self.south + (self.elevations.shape[0] - 1) * self.arc_seconds / 3600

    @property
    def east(self) -> float:
        if self.raster_bounds is not None:
            return self.raster_bounds["east"]
        return self.west + (self.elevations.shape[1] - 1) * self.arc_seconds / 3600

    @property
    def bounds(self) -> dict[str, float]:
        return {"south": self.south, "west": self.west,
                "north": self.north, "east": self.east}

    @property
    def resolution_m(self) -> tuple[float, float]:
        if self.pixel_size_m is not None:
            return self.pixel_size_m
        spacing = self.arc_seconds / 3600 * 111_000
        return spacing, spacing

    @property
    def statistics(self) -> dict[str, float | int | None]:
        valid = self.elevations[np.isfinite(self.elevations)]
        if not valid.size:
            return {"minimum_m": None, "maximum_m": None, "mean_m": None,
                    "valid_pixels": 0}
        return {"minimum_m": round(float(valid.min()), 2),
                "maximum_m": round(float(valid.max()), 2),
                "mean_m": round(float(valid.mean()), 2),
                "valid_pixels": int(valid.size)}

    def _world_to_pixel(self, lat: float, lon: float) -> tuple[float, float]:
        if self.geo_transform is None:
            spacing = self.arc_seconds / 3600
            return (lon - self.west) / spacing, (self.north - lat) / spacing

        x, y = lon, lat
        if self.crs and self.crs.upper() not in {"EPSG:4326", "OGC:CRS84", "WGS84"}:
            try:
                from rasterio.warp import transform
                x_values, y_values = transform("EPSG:4326", self.crs, [lon], [lat])
                x, y = x_values[0], y_values[0]
            except ImportError as exc:
                raise RuntimeError("rasterio is required to sample a projected DEM") from exc
        a, b, c, d, e, f = self.geo_transform
        determinant = a * e - b * d
        if abs(determinant) < 1e-12:
            raise ValueError("DEM geotransform is not invertible")
        dx, dy = x - c, y - f
        return ((e * dx - b * dy) / determinant,
                (-d * dx + a * dy) / determinant)

    def sample(self, lat: float, lon: float) -> float:
        if not self.south <= lat <= self.north or not self.west <= lon <= self.east:
            raise ValueError("coordinate is outside DEM bounds")
        x, y = self._world_to_pixel(lat, lon)
        if x < -1 or y < -1 or x > self.elevations.shape[1] or y > self.elevations.shape[0]:
            raise ValueError("coordinate is outside DEM raster")
        x = float(np.clip(x, 0, self.elevations.shape[1] - 1))
        y = float(np.clip(y, 0, self.elevations.shape[0] - 1))
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, self.elevations.shape[1] - 1), min(y0 + 1, self.elevations.shape[0] - 1)
        dx, dy = x - x0, y - y0
        values = np.asarray([[self.elevations[y0, x0], self.elevations[y0, x1]],
                             [self.elevations[y1, x0], self.elevations[y1, x1]]], dtype=np.float64)
        weights = np.asarray([[(1 - dx) * (1 - dy), dx * (1 - dy)],
                              [(1 - dx) * dy, dx * dy]])
        valid = np.isfinite(values)
        if not valid.any():
            raise ValueError("coordinate falls on DEM nodata")
        return float((values[valid] * weights[valid]).sum() / weights[valid].sum())

    def slope_degrees(self, lat: float, lon: float) -> float:
        """Estimate local slope using a central finite difference in metres."""
        spacing_x, spacing_y = self.resolution_m
        step_lat = max(spacing_y / 111_000, 1e-7)
        step_lon = max(spacing_x / (111_000 * max(cos(radians(lat)), .1)), 1e-7)
        try:
            west = self.sample(lat, lon - step_lon)
            east = self.sample(lat, lon + step_lon)
            south = self.sample(lat - step_lat, lon)
            north = self.sample(lat + step_lat, lon)
        except ValueError:
            return 0.0
        dz_dx = (east - west) / max(2 * spacing_x, 1e-6)
        dz_dy = (north - south) / max(2 * spacing_y, 1e-6)
        return float(np.degrees(np.arctan(np.hypot(dz_dx, dz_dy))))

    def cell_features(self, lat: float, lon: float) -> dict[str, float]:
        elevation = self.sample(lat, lon)
        slope = self.slope_degrees(lat, lon)
        return {"elevation_m": round(elevation, 2), "slope_deg": round(slope, 2),
                "slope_score": round(float(np.clip(1 - slope / 45, 0, 1)), 3)}


def load_hgt(path: str | Path) -> ElevationGrid:
    path = Path(path)
    size = path.stat().st_size
    side = 3601 if size == 3601 * 3601 * 2 else 1201 if size == 1201 * 1201 * 2 else 0
    if not side:
        raise ValueError("DEM must be a 1-arcsecond or 3-arcsecond SRTM HGT file")
    stem = path.stem.upper()
    if not (stem.startswith("N") or stem.startswith("S")) or "E" not in stem and "W" not in stem:
        raise ValueError("HGT filename must encode tile bounds, for example N30E078.hgt")
    lat_part, lon_part = stem[1:].split("E") if "E" in stem else stem[1:].split("W")
    south = float(lat_part) * (1 if stem[0] == "N" else -1)
    west = float(lon_part) * (1 if "E" in stem else -1)
    values = np.fromfile(path, dtype=">i2").reshape(side, side).astype(np.float32)
    values[values == -32768] = np.nan
    arc_seconds = 1 if side == 3601 else 3
    return ElevationGrid(values, south, west, arc_seconds,
                         source="srtm", nodata=-32768)


def load_geotiff(path: str | Path) -> ElevationGrid:
    """Load a single-band GeoTIFF/COG DEM while retaining its native transform."""
    try:
        import rasterio
        from rasterio.warp import transform_bounds
    except ImportError as exc:
        raise RuntimeError("rasterio is required for GeoTIFF/COG DEMs; install requirements.txt") from exc

    path = Path(path)
    with rasterio.open(path) as dataset:
        if dataset.count < 1:
            raise ValueError("DEM GeoTIFF contains no raster bands")
        if dataset.crs is None:
            raise ValueError("DEM GeoTIFF must include a CRS")
        values = dataset.read(1).astype(np.float32)
        nodata = dataset.nodata
        if nodata is not None:
            values[values == nodata] = np.nan
        native_bounds = dataset.bounds
        crs = str(dataset.crs)
        if crs.upper() in {"EPSG:4326", "OGC:CRS84", "WGS84"}:
            bounds = native_bounds
        else:
            west, south, east, north = transform_bounds(crs, "EPSG:4326", *native_bounds, densify_pts=21)
            bounds = type(native_bounds)(west, south, east, north)
        transform_obj = dataset.transform
        geo_transform = (transform_obj.a, transform_obj.b, transform_obj.c,
                         transform_obj.d, transform_obj.e, transform_obj.f)
        center_lat = (bounds.bottom + bounds.top) / 2
        if dataset.crs.is_projected:
            pixel_size_m = (abs(float(transform_obj.a)), abs(float(transform_obj.e)))
        else:
            pixel_size_m = (abs(float(transform_obj.a)) * 111_000 * max(cos(radians(center_lat)), .1),
                            abs(float(transform_obj.e)) * 111_000)
        tags = dataset.tags()
        vertical_units = str(tags.get("vertical_units", tags.get("units", "m")))
        arc_seconds = max(abs(float(transform_obj.e)), 1e-9) * 3600
        return ElevationGrid(values, float(bounds.bottom), float(bounds.left), arc_seconds,
                             crs=crs, geo_transform=geo_transform,
                             raster_bounds={"south": float(bounds.bottom), "west": float(bounds.left),
                                            "north": float(bounds.top), "east": float(bounds.right)},
                             pixel_size_m=pixel_size_m, vertical_units=vertical_units,
                             source="geotiff", nodata=nodata)


def load_dem(path: str | Path) -> ElevationGrid:
    """Load a supported local DEM format."""
    suffix = Path(path).suffix.lower()
    if suffix == ".hgt":
        return load_hgt(path)
    if suffix in {".tif", ".tiff", ".cog"}:
        return load_geotiff(path)
    raise ValueError("DEM must be an SRTM .hgt or GeoTIFF/COG file")
