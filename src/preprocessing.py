import pandas as pd

def prepare_arima_series(df, drug_name, test_size=30):
    """
    Prepares a drug's time series for ARIMA.
    Returns train and test splits with datetime index.
    """
    series = df.set_index("datum")[drug_name]
    series.index.freq = "D"

    train = series[:-test_size]
    test = series[-test_size:]

    return train, test

def prepare_prophet_df(df, drug_name, test_size=30):
    """
    Prepares a drug's dataframe for Prophet.
    Returns train and test splits with ds/y column names.
    """
    prophet_df = df[["datum", drug_name]].rename(columns={"datum": "ds", drug_name: "y"})

    train = prophet_df[:-test_size]
    test = prophet_df[-test_size:]

    return train, test