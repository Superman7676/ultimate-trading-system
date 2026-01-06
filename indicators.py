#!/usr/bin/env python3
"""
Ultimate Trading System - Technical Indicators Module
Implementation of popular technical analysis indicators
"""

import pandas as pd
import numpy as np
from typing import Tuple


class TechnicalIndicators:
    """
    Class containing technical analysis indicators
    All methods are static and work with pandas DataFrames
    """
    
    @staticmethod
    def simple_moving_average(data: pd.Series, period: int = 20) -> pd.Series:
        """
        Simple Moving Average (SMA)
        
        Args:
            data: Price series (usually Close prices)
            period: Number of periods for the average
        
        Returns:
            SMA values
        """
        return data.rolling(window=period).mean()
    
    @staticmethod
    def exponential_moving_average(data: pd.Series, period: int = 12) -> pd.Series:
        """
        Exponential Moving Average (EMA)
        
        Args:
            data: Price series
            period: Number of periods
        
        Returns:
            EMA values
        """
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def relative_strength_index(data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI)
        Measures momentum and identifies overbought/oversold conditions
        
        Args:
            data: Price series
            period: Number of periods (default 14)
        
        Returns:
            RSI values (0-100)
        """
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def moving_average_convergence_divergence(
        data: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence (MACD)
        
        Args:
            data: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
        
        Returns:
            Tuple of (MACD, Signal Line, Histogram)
        """
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return macd, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(
        data: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands
        Shows support/resistance and volatility
        
        Args:
            data: Price series
            period: Period for moving average
            std_dev: Number of standard deviations
        
        Returns:
            Tuple of (Middle Band, Upper Band, Lower Band)
        """
        middle_band = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return middle_band, upper_band, lower_band
    
    @staticmethod
    def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average True Range (ATR)
        Measures volatility
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Period for ATR
        
        Returns:
            ATR values
        """
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator
        Identifies overbought/oversold conditions
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: Stochastic period
            smooth_k: K smoothing period
            smooth_d: D smoothing period
        
        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        k_percent_smooth = k_percent.rolling(window=smooth_k).mean()
        d_percent = k_percent_smooth.rolling(window=smooth_d).mean()
        
        return k_percent_smooth, d_percent
    
    @staticmethod
    def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On Balance Volume (OBV)
        Shows money flow and volume strength
        
        Args:
            close: Close prices
            volume: Volume data
        
        Returns:
            OBV values
        """
        obv = volume.copy()
        obv[close.diff() < 0] = -obv[close.diff() < 0]
        obv[close.diff() == 0] = 0
        
        return obv.cumsum()
    
    @staticmethod
    def accumulation_distribution_line(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Accumulation/Distribution Line (ADL)
        Measures money flow
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume data
        
        Returns:
            ADL values
        """
        clv = ((close - low) - (high - close)) / (high - low)
        ad = (clv * volume).cumsum()
        
        return ad
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a given DataFrame
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with all indicators added
        """
        # Moving Averages
        df['SMA_20'] = TechnicalIndicators.simple_moving_average(df['Close'], 20)
        df['SMA_50'] = TechnicalIndicators.simple_moving_average(df['Close'], 50)
        df['SMA_200'] = TechnicalIndicators.simple_moving_average(df['Close'], 200)
        
        df['EMA_12'] = TechnicalIndicators.exponential_moving_average(df['Close'], 12)
        df['EMA_26'] = TechnicalIndicators.exponential_moving_average(df['Close'], 26)
        
        # Momentum
        df['RSI'] = TechnicalIndicators.relative_strength_index(df['Close'], 14)
        
        macd, signal, histogram = TechnicalIndicators.moving_average_convergence_divergence(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Histogram'] = histogram
        
        # Volatility
        middle, upper, lower = TechnicalIndicators.bollinger_bands(df['Close'], 20, 2.0)
        df['BB_Middle'] = middle
        df['BB_Upper'] = upper
        df['BB_Lower'] = lower
        
        df['ATR'] = TechnicalIndicators.average_true_range(df['High'], df['Low'], df['Close'], 14)
        
        # Volume
        df['OBV'] = TechnicalIndicators.on_balance_volume(df['Close'], df['Volume'])
        df['ADL'] = TechnicalIndicators.accumulation_distribution_line(
            df['High'], df['Low'], df['Close'], df['Volume']
        )
        
        # Stochastic
        k_percent, d_percent = TechnicalIndicators.stochastic(
            df['High'], df['Low'], df['Close']
        )
        df['Stochastic_K'] = k_percent
        df['Stochastic_D'] = d_percent
        
        return df


if __name__ == '__main__':
    # Example usage
    print("Technical Indicators Module")
    print("Import this module to use technical indicators")
    print("\nExample:")
    print("from indicators import TechnicalIndicators")
    print("df = TechnicalIndicators.calculate_all_indicators(df)")
