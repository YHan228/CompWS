"""
External data source fetchers for noise/diffusion coefficients.

See IMPLEMENTATION_PLAN.md Section 2.1 for full specification.
"""

import numpy as np
import pandas as pd
import requests_cache
import yfinance as yf
from openmeteo_requests import Client


class VolatilityFetcher:
    """
    Fetch historical financial volatility (e.g., VIX) as noise source.

    Uses yfinance to retrieve daily data.
    """

    def __init__(
        self, ticker: str = "^VIX", start: str = "2014-01-01", end: str = "2024-01-01"
    ):
        """
        Args:
            ticker: Yahoo Finance ticker symbol (default: VIX index)
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
        """
        self.ticker = ticker
        self.start = start
        self.end = end
        self._series: pd.Series | None = None

    def fetch(self) -> pd.Series:
        """
        Fetch daily closing values.

        Returns:
            pandas Series with datetime index and daily values
        """
        data = yf.download(self.ticker, start=self.start, end=self.end, progress=False)
        if data.empty:
            raise ValueError("No data returned from yfinance for given range.")
        series = data["Close"].dropna()
        self._series = series
        return series

    def to_diffusion_coefficients(self, scale: float = 0.01) -> np.ndarray:
        """
        Normalize fetched data to [0, scale] range for use as sigma(t) in SDE.

        Args:
            scale: Maximum diffusion coefficient value

        Returns:
            numpy array of normalized values, shape (T,)
        """
        series = self._series if self._series is not None else self.fetch()
        values = series.to_numpy(dtype=float)
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        if np.isclose(max_val, min_val):
            return np.zeros_like(values, dtype=float)
        normalized = (values - min_val) / (max_val - min_val)
        return normalized * scale


class WeatherFetcher:
    """
    Fetch historical weather data (wind speed, temperature, etc.) as noise source.

    Uses Open-Meteo API.
    """

    def __init__(
        self,
        latitude: float = 35.6762,  # Tokyo default
        longitude: float = 139.6503,
        start: str = "2014-01-01",
        end: str = "2024-01-01",
    ):
        """
        Args:
            latitude: Location latitude
            longitude: Location longitude
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
        """
        self.latitude = latitude
        self.longitude = longitude
        self.start = start
        self.end = end
        self._series: pd.Series | None = None

    def fetch(self, variable: str = "wind_speed_10m_max") -> pd.Series:
        """
        Fetch daily weather variable.

        Args:
            variable: Open-Meteo variable name (e.g., "wind_speed_10m_max",
                     "temperature_2m_max", "precipitation_sum")

        Returns:
            pandas Series with datetime index and daily values
        """
        url = "https://archive-api.open-meteo.com/v1/archive"
        session = requests_cache.CachedSession(".openmeteo_cache", expire_after=3600)
        client = Client(session=session)
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": self.start,
            "end_date": self.end,
            "daily": variable,
            "timezone": "UTC",
        }
        responses = client.weather_api(url, params=params)
        response = responses[0]
        daily = response.Daily()
        values = daily.Variables(0).ValuesAsNumpy()
        times = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        )
        series = pd.Series(values, index=times)
        self._series = series
        return series

    def to_diffusion_coefficients(self, scale: float = 0.01) -> np.ndarray:
        """
        Normalize fetched data to [0, scale] range for use as sigma(t) in SDE.

        Args:
            scale: Maximum diffusion coefficient value

        Returns:
            numpy array of normalized values, shape (T,)
        """
        series = self._series if self._series is not None else self.fetch()
        values = series.to_numpy(dtype=float)
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        if np.isclose(max_val, min_val):
            return np.zeros_like(values, dtype=float)
        normalized = (values - min_val) / (max_val - min_val)
        return normalized * scale
