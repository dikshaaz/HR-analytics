# DataProcessing.py
import pandas as pd

def load_and_process_data():
    """
    Loads employee.csv dataset and cleans it.
    Returns cleaned DataFrame.
    """
    path = "employee.csv"
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = df.columns.str.strip().str.replace(" ", "_")

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Fill missing numeric values with median
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Fill missing categorical values with 'Unknown'
    df.fillna('Unknown', inplace=True)

    print("✔ Data loaded and cleaned successfully!")
    print(f"Dataset shape: {df.shape}")

    return df