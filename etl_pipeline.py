import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
import kagglehub

def run_etl():
    """
    Runs the full ETL pipeline:
    1. Loads raw data.
    2. Performs feature engineering.
    3. Saves processed data to a PostgreSQL database.
    """
    print("Starting ETL pipeline...")

    # 1. EXTRACT
    print("\n[1/3] Loading raw data...")
    try:
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        print(f"Path to dataset files: {path}")
        file = os.listdir(path)
        print(f"Files in dataset folder: {file}")
        df = pd.read_csv(os.path.join(path, file[0]))
        print(f"Raw data shape: {df.shape}")
        print("Raw data sample (first 5 rows):")
        print(df.head())
    except FileNotFoundError:
        print("Error: creditcard.csv not found. Download or edit the code.")
        return

    # 2. TRANSFORM
    print("\n[2/3] Performing feature engineering...")

    print("Adding time-based features...")
    df['Hour'] = (df['Time'] // 3600) % 24
    df['DayOfWeek'] = (df['Time'] // (3600 * 24)) % 7

    print("Simulating CardID feature...")
    num_cards = 10000
    df['CardID'] = np.random.randint(0, num_cards, size=len(df))

    print("Sorting data by CardID and Time...")
    df = df.sort_values(by=['CardID', 'Time'])

    print("Calculating rolling average transaction amount...")
    df['Avg_Amount'] = df.groupby('CardID')['Amount'].transform(
        lambda x: x.rolling(window=10, min_periods=1).mean().shift(1)
    )
    df['Avg_Amount'] = df['Avg_Amount'].fillna(df['Amount'])
    df['Amount_to_Avg_Amount'] = df['Amount'] / (df['Avg_Amount'] + 1e-6)

    print("Calculating transaction frequency (time difference)...")
    df['Trans_Freq'] = df.groupby('CardID')['Time'].diff().fillna(0)

    print("Dropping original 'Time' column...")
    df.drop('Time', axis=1, inplace=True)

    print(f"Transformation complete. Data shape: {df.shape}")
    print("Processed data sample (first 5 rows):")
    print(df.head())

    # 3. LOAD
    print("\n[3/3] Saving processed data to database...")

    DB_USER = os.getenv("DB_USER", "postgres")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "fraud_detection")

    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    try:
        engine = create_engine(DATABASE_URL)
        df.to_sql('processed_transactions', engine, if_exists='replace', index=False, chunksize=10000)
        print("Processed data successfully saved to 'processed_transactions' table.")
        print("ETL pipeline finished successfully.")
    except Exception as e:
        print(f"Error connecting to or writing to the database: {e}")

if __name__ == "__main__":
    run_etl()