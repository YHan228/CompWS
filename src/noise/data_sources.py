"""
External data source fetchers for noise/diffusion coefficients.

See IMPLEMENTATION_PLAN.md Section 2.1 for full specification.
"""

import numpy as np
import pandas as pd


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
        raise NotImplementedError

    def fetch(self) -> pd.Series:
        """
        Fetch daily closing values.

        Returns:
            pandas Series with datetime index and daily values
        """
        raise NotImplementedError

    def to_diffusion_coefficients(self, scale: float = 0.01) -> np.ndarray:
        """
        Normalize fetched data to [0, scale] range for use as sigma(t) in SDE.

        Args:
            scale: Maximum diffusion coefficient value

        Returns:
            numpy array of normalized values, shape (T,)
        """
        raise NotImplementedError


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
        raise NotImplementedError

    def fetch(self, variable: str = "wind_speed_10m_max") -> pd.Series:
        """
        Fetch daily weather variable.

        Args:
            variable: Open-Meteo variable name (e.g., "wind_speed_10m_max",
                     "temperature_2m_max", "precipitation_sum")

        Returns:
            pandas Series with datetime index and daily values
        """
        raise NotImplementedError

    def to_diffusion_coefficients(self, scale: float = 0.01) -> np.ndarray:
        """
        Normalize fetched data to [0, scale] range for use as sigma(t) in SDE.

        Args:
            scale: Maximum diffusion coefficient value

        Returns:
            numpy array of normalized values, shape (T,)
        """
        raise NotImplementedError
