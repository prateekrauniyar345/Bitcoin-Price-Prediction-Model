
import pandas as pd

def load_and_prepare_data(raw_data_path, processed_data_path):
    """
    Loads raw data, cleans it, engineers features, and saves the processed data.
    """
    # Load the raw data
    df = pd.read_csv(raw_data_path)

    # Drop rows with missing values
    df = df.dropna()

    # Convert timestamp to datetime objects
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')

    # Feature Engineering (example: create a moving average)
    df['MA30'] = df['Close'].rolling(window=30).mean()

    # Drop rows with NaN values created by the rolling mean
    df = df.dropna()

    # Save the processed data
    df.to_csv(processed_data_path, index=False)

if __name__ == '__main__':
    # Define file paths
    raw_data_path = 'data/btcusd_1-min_data.csv'
    processed_data_path = 'data/processed/processed_data.csv'

    # Run the data preparation
    load_and_prepare_data(raw_data_path, processed_data_path)

    print("Data preparation complete. Processed data saved to:", processed_data_path)

