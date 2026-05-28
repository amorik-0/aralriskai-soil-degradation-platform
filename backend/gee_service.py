"""
gee_service.py — Google Earth Engine Data Extraction
=====================================================
Extracts Sentinel-2, ERA5 and MODIS data for a given location and date range.

Usage:
  import ee; ee.Initialize()
  data = GEEService.get_point_features(lat=44.5, lon=59.2, date='2024-06-01')

Datasets:
  - Sentinel-2 SR Harmonized : COPERNICUS/S2_SR_HARMONIZED
  - ERA5-Land Hourly          : ECMWF/ERA5_LAND/HOURLY
  - ERA5 Hourly (wind)        : ECMWF/ERA5/HOURLY
  - MODIS Land Cover          : MODIS/006/MCD12Q1
"""

import ee
import math
import datetime
from typing import Optional, Dict


class GEEService:
    """Handles all Google Earth Engine data fetching."""

    SENTINEL2 = "COPERNICUS/S2_SR_HARMONIZED"
    ERA5_LAND  = "ECMWF/ERA5_LAND/HOURLY"
    ERA5_WIND  = "ECMWF/ERA5/HOURLY"
    MODIS_LC   = "MODIS/006/MCD12Q1"

    @staticmethod
    def _cloud_mask_s2(image: ee.Image) -> ee.Image:
        """
        Apply Sentinel-2 cloud mask using the QA60 band.
        Bits 10 and 11 encode opaque and cirrus clouds.
        Source: GEE Sentinel-2 documentation
        """
        qa = image.select("QA60")
        cloud_bit_mask  = 1 << 10  # opaque clouds
        cirrus_bit_mask = 1 << 11  # cirrus clouds
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
               qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        return image.updateMask(mask).divide(10000)  # scale reflectance to 0-1

    @staticmethod
    def get_sentinel2_image(lat: float, lon: float,
                            start_date: str, end_date: str) -> ee.Image:
        """
        Fetch cloud-free Sentinel-2 SR composite for a location and period.

        Args:
            lat, lon   : WGS84 coordinates
            start_date : ISO date string, e.g. '2024-05-01'
            end_date   : ISO date string, e.g. '2024-06-01'

        Returns:
            ee.Image with bands B2 (BLUE), B3 (GREEN), B4 (RED),
                                B8 (NIR), B11 (SWIR-1), B12 (SWIR-2)
        """
        point = ee.Geometry.Point([lon, lat])
        col = (
            ee.ImageCollection(GEEService.SENTINEL2)
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .map(GEEService._cloud_mask_s2)
            .select(["B2", "B3", "B4", "B8", "B11", "B12"])
        )
        return col.median()  # median composite reduces residual cloud noise

    @staticmethod
    def get_era5_climate(lat: float, lon: float,
                         start_date: str, end_date: str) -> Dict:
        """
        Extract ERA5-Land climate variables for a location and period.

        Variables extracted:
          - temperature_2m       : 2m air temperature (K → °C)
          - soil_water_layer_1   : volumetric soil water layer 1 (m³/m³)
          - total_precipitation  : accumulated precipitation (m)
          - u_component_of_wind_10m : u wind component (m/s)  [ERA5, not ERA5-Land]
          - v_component_of_wind_10m : v wind component (m/s)

        Returns:
            dict with mean values for the period
        """
        point = ee.Geometry.Point([lon, lat])

        # ERA5-Land for soil moisture, temperature, precipitation
        era5_land = (
            ee.ImageCollection(GEEService.ERA5_LAND)
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .select(["temperature_2m", "soil_water_layer_1", "total_precipitation"])
            .mean()
        )

        # ERA5 for wind components (ERA5-Land has coarser wind data)
        era5_wind = (
            ee.ImageCollection(GEEService.ERA5_WIND)
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
            .mean()
        )

        land_vals = era5_land.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=11132  # ERA5-Land native resolution (~0.1°)
        ).getInfo()

        wind_vals = era5_wind.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=27830  # ERA5 native resolution (~0.25°)
        ).getInfo()

        u = wind_vals.get("u_component_of_wind_10m", 0) or 0
        v = wind_vals.get("v_component_of_wind_10m", 0) or 0

        return {
            "temperature_c": (land_vals.get("temperature_2m", 273.15) - 273.15),
            "soil_moisture":  land_vals.get("soil_water_layer_1", 0.1),
            "precipitation":  land_vals.get("total_precipitation", 0),
            # Wind speed from components: speed = √(u² + v²)
            "wind_speed":     math.sqrt(u**2 + v**2),
            # Wind direction (meteorological): atan2(u, v) → degrees
            "wind_direction": (math.degrees(math.atan2(u, v)) + 360) % 360,
            "u_wind": u,
            "v_wind": v,
        }

    @staticmethod
    def get_land_cover(lat: float, lon: float, year: int = 2023) -> int:
        """
        Fetch MODIS land cover type (IGBP classification) for a location.
        LC_Type1 values:
          0 = Water, 1 = Forest, 10 = Grasslands,
          12 = Croplands, 16 = Barren, 17 = Urban
        """
        point = ee.Geometry.Point([lon, lat])
        lc = (
            ee.ImageCollection(GEEService.MODIS_LC)
            .filter(ee.Filter.calendarRange(year, year, "year"))
            .first()
            .select("LC_Type1")
        )
        val = lc.reduceRegion(
            reducer=ee.Reducer.first(), geometry=point, scale=500
        ).getInfo()
        return val.get("LC_Type1", -1)

    @staticmethod
    def extract_band_values(image: ee.Image, lat: float, lon: float) -> Dict:
        """
        Extract raw Sentinel-2 band values at a point (for index computation).
        Returns reflectance values scaled 0–1 after divide(10000).
        """
        point = ee.Geometry.Point([lon, lat])
        vals = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10  # Sentinel-2 native 10m resolution
        ).getInfo()
        return {
            "B2_blue":  vals.get("B2",  0),
            "B3_green": vals.get("B3",  0),
            "B4_red":   vals.get("B4",  0),
            "B8_nir":   vals.get("B8",  0),
            "B11_swir": vals.get("B11", 0),
            "B12_swir2":vals.get("B12", 0),
        }

    @staticmethod
    def get_ndvi_time_series(lat: float, lon: float,
                             start_date: str, end_date: str) -> list:
        """
        Returns monthly NDVI values for trend analysis and anomaly detection.
        """
        point = ee.Geometry.Point([lon, lat])
        col = (
            ee.ImageCollection(GEEService.SENTINEL2)
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .map(GEEService._cloud_mask_s2)
        )

        def compute_ndvi(img):
            ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
            return ndvi.set("system:time_start", img.get("system:time_start"))

        ndvi_col = col.map(compute_ndvi)
        ts = ndvi_col.reduceRegion(
            reducer=ee.Reducer.mean().setOutputs(["NDVI"]),
            geometry=point,
            scale=10
        )
        # For production: use getRegion() for full time series
        return ndvi_col.aggregate_array("NDVI").getInfo()
