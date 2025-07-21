import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands
import yfinance as yf
import logging
from typing import Optional, Tuple
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPreparation:
    """Handles data preparation for XAUUSD time series forecasting.

    Computes 10 features per time step and saves the processed data as a CSV.
    """

    FEATURES: list[str] = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA', 'RSI', 'MACD', 'BB_Width', 'USD_Index'
    ]
    OUTPUT_CSV: Path = Path('xauusd_features.csv')

    def __init__(self, output_dir: str = '.'):
        """Initializes the DataPreparation class.

        Args:
            output_dir (str): Directory to save the output CSV file.
        """
        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path: Path = self.output_dir / self.OUTPUT_CSV

    def fetch_data(
        self,
        start_date: str = '2020-01-01',
        end_date: str = '2025-07-22',
        interval: str = '1d'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetches XAUUSD and USD Index data using yfinance.

        Args:
            start_date (str): Start date for data fetching (YYYY-MM-DD).
            end_date (str): End date for data fetching (YYYY-MM-DD).
            interval (str): Data interval (e.g., '1d' for daily).

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: XAUUSD and USD Index data.

        Raises:
            RuntimeError: If data fetching fails.
        """
        try:
            logging.info("Fetching XAUUSD data...")
            xauusd: pd.DataFrame = yf.download('GC=F', start=start_date, end=end_date, interval=interval, auto_adjust=False)
            logging.info("Fetching USD Index data...")
            usd_index: pd.DataFrame = yf.download('DX-Y.NYB', start=start_date, end=end_date, interval=interval, auto_adjust=False)
            
            if xauusd.empty or usd_index.empty:
                logging.error("Failed to fetch data: empty DataFrame returned")
                raise RuntimeError("Failed to fetch XAUUSD or USD Index data")
            
            return xauusd, usd_index
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            raise RuntimeError(f"Failed to fetch data: {e}") from e

    def compute_features(self, xauusd: pd.DataFrame, usd_index: pd.DataFrame) -> pd.DataFrame:
        """Computes technical indicators and combines features.

        Args:
            xauusd (pd.DataFrame): XAUUSD OHLCV data.
            usd_index (pd.DataFrame): USD Index data.

        Returns:
            pd.DataFrame: DataFrame with 10 features per time step.

        Raises:
            ValueError: If input DataFrames are incompatible.
        """
        if xauusd.index.tz != usd_index.index.tz:
            logging.error("Mismatched time zones in input DataFrames")
            raise ValueError("XAUUSD and USD Index DataFrames must have the same time zone")
        
        data: pd.DataFrame = xauusd[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        data['USD_Index'] = usd_index['Close']
        
        # Ensure Close is a Series
        close_series: pd.Series = data['Close'].squeeze()
        
        # Compute technical indicators
        logging.info("Computing technical indicators...")
        data['SMA'] = SMAIndicator(close_series, window=10).sma_indicator()
        data['RSI'] = RSIIndicator(close_series, window=14).rsi()
        data['MACD'] = MACD(close_series).macd()
        bb: BollingerBands = BollingerBands(close_series, window=20)
        data['BB_Width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        
        # Drop rows with NaN values
        data = data.dropna()
        if data.empty:
            logging.error("No data remains after dropping NaN values")
            raise ValueError("Computed features resulted in an empty DataFrame")
        
        return data[self.FEATURES]

    def normalize_features(self, data: pd.DataFrame) -> np.ndarray:
        """Normalizes features using MinMaxScaler.

        Args:
            data (pd.DataFrame): DataFrame with features to normalize.

        Returns:
            np.ndarray: Normalized feature array of shape (num_timesteps, 10).

        Raises:
            ValueError: If input DataFrame does not contain all required features.
        """
        if not all(feature in data.columns for feature in self.FEATURES):
            missing: list[str] = [f for f in self.FEATURES if f not in data.columns]
            logging.error(f"Missing features in DataFrame: {missing}")
            raise ValueError(f"DataFrame missing features: {missing}")
        
        scaler: MinMaxScaler = MinMaxScaler()
        normalized_data: np.ndarray = scaler.fit_transform(data)
        logging.info("Features normalized successfully")
        return normalized_data

    def save_to_csv(self, data: pd.DataFrame) -> None:
        """Saves the DataFrame to a CSV file.

        Args:
            data (pd.DataFrame): DataFrame to save.

        Raises:
            RuntimeError: If saving to CSV fails.
        """
        try:
            data.to_csv(self.output_path, index=True)
            logging.info(f"Saved data to {self.output_path}")
        except Exception as e:
            logging.error(f"Failed to save CSV: {e}")
            raise RuntimeError(f"Failed to save CSV: {e}") from e

    def prepare_data(
        self,
        start_date: str = '2020-01-01',
        end_date: str = '2025-07-22',
        interval: str = '1d',
        force_generate: bool = False
    ) -> np.ndarray:
        """Prepares XAUUSD time series data with 10 features per time step.

        Args:
            start_date (str): Start date for data fetching.
            end_date (str): End date for data fetching.
            interval (str): Data interval (e.g., '1d' for daily).
            force_generate (bool): If True, regenerates CSV even if it exists.

        Returns:
            np.ndarray: Normalized feature array of shape (num_timesteps, 10).

        Raises:
            RuntimeError: If data preparation fails.
        """
        if self.output_path.exists() and not force_generate:
            logging.info(f"Loading existing data from {self.output_path}")
            try:
                data: pd.DataFrame = pd.read_csv(self.output_path, index_col=0)
                return self.normalize_features(data)
            except Exception as e:
                logging.warning(f"Failed to load existing CSV: {e}. Regenerating data...")
        
        xauusd, usd_index = self.fetch_data(start_date, end_date, interval)
        feature_data: pd.DataFrame = self.compute_features(xauusd, usd_index)
        self.save_to_csv(feature_data)
        return self.normalize_features(feature_data)

def main():
    """Main function to run data preparation from the command line."""
    parser = argparse.ArgumentParser(description="Prepare XAUUSD time series data for forecasting.")
    parser.add_argument('--start-date', default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2025-07-22', help='End date (YYYY-MM-DD)')
    parser.add_argument('--interval', default='1d', help='Data interval (e.g., 1d, 1h)')
    parser.add_argument('--force', action='store_true', help='Force regenerate CSV')
    parser.add_argument('--output-dir', default='.', help='Directory to save output CSV')
    args = parser.parse_args()

    try:
        prep: DataPreparation = DataPreparation(output_dir=args.output_dir)
        normalized_data: np.ndarray = prep.prepare_data(
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval,
            force_generate=args.force
        )
        logging.info(f"Prepared data with shape: {normalized_data.shape}")
    except Exception as e:
        logging.error(f"Data preparation failed: {e}")
        raise

if __name__ == "__main__":
    main()
