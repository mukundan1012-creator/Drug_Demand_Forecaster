import numpy as np
import joblib
import os
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error
from src.preprocessing import prepare_arima_series, prepare_prophet_df  

def forecast_drug(drug_name, df, test_size=30):
    """
    Trains ARIMA and Prophet on a drug category,
    compares RMSE, picks winner, returns results + trained models.
    """
    
    train_arima, test_arima = prepare_arima_series(df, drug_name, test_size) 

    arima_model = ARIMA(train_arima, order=(2, 0, 2))
    arima_fit = arima_model.fit()
    arima_forecast = arima_fit.forecast(steps=test_size)
    arima_forecast.index = test_arima.index
    arima_rmse = np.sqrt(mean_squared_error(test_arima, arima_forecast))

  
    train_prophet, test_prophet = prepare_prophet_df(df, drug_name, test_size)  

    prophet_model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    prophet_model.fit(train_prophet)
    future = prophet_model.make_future_dataframe(periods=test_size)
    prophet_forecast = prophet_model.predict(future)["yhat"].tail(test_size).values
    prophet_rmse = np.sqrt(mean_squared_error(test_prophet["y"].values, prophet_forecast))


    if arima_rmse <= prophet_rmse:
        winner = "ARIMA"
        best_forecast = arima_forecast.values
        best_rmse = arima_rmse
    else:
        winner = "Prophet"
        best_forecast = prophet_forecast
        best_rmse = prophet_rmse

    return {
        "drug": drug_name,
        "arima_rmse": round(arima_rmse, 3),
        "prophet_rmse": round(prophet_rmse, 3),
        "winner": winner,
        "best_rmse": round(best_rmse, 3),
        "test_actual": test_arima.values,
        "best_forecast": best_forecast,
        "test_dates": test_arima.index,
        "arima_model": arima_fit,              
        "prophet_model": prophet_model           
    }


def save_models(results):
    """
    Saves all 16 trained models (ARIMA + Prophet per drug) to Model/ folder.
    """
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
    model_dir = os.path.join(BASE_DIR, "Model")
    os.makedirs(model_dir, exist_ok=True)                                    

    for r in results:
        joblib.dump(r["arima_model"], os.path.join(model_dir, f"{r['drug']}_arima.pkl"))    
        joblib.dump(r["prophet_model"], os.path.join(model_dir, f"{r['drug']}_prophet.pkl")) 
        print(f"Saved models for {r['drug']}")