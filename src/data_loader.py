import pandas as pd
import os

def load_data():
    """
    Loads and returns the cleaned pharma sales dataframe.
    Converts datum to datetime.
    """   
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "salesdaily.csv")

    df = pd.read_csv(data_path)
    df["datum"] = pd.to_datetime(df["datum"])

    return df

DRUG_COLUMNS = ["M01AB", "M01AE", "N02BA", "N02BE", "N05B", "N05C", "R03", "R06"] 