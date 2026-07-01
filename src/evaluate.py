import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_metrics(actual, forecast):
    """
    Computes RMSE and MAPE given actual and forecasted values.
    MAPE excludes zero-actual days to avoid division by zero.
    """
    rmse = np.sqrt(mean_squared_error(actual, forecast))          

    nonzero_mask = actual != 0                                     
    mape = np.mean(
        np.abs((actual[nonzero_mask] - forecast[nonzero_mask]) /
               actual[nonzero_mask])
    ) * 100                                                        

    return round(rmse, 3), round(mape, 3)


def save_summary_csv(results):
    """
    Saves model comparison summary table as CSV to project root.
    """
    summary = pd.DataFrame([{
        "Drug": r["drug"],
        "ARIMA RMSE": r["arima_rmse"],
        "Prophet RMSE": r["prophet_rmse"],
        "Winner": r["winner"],
        "Best RMSE": r["best_rmse"]
    } for r in results])

    save_path = os.path.join(BASE_DIR, "model_comparison.csv")    
    summary.to_csv(save_path, index=False)
    print(f"Summary saved to {save_path}")

    return summary


def save_forecast_plot(results):
    """
    Saves 8-panel forecast vs actual plot for all drug categories.
    """
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))              
    axes = axes.flatten()                                          

    for i, r in enumerate(results):
        axes[i].plot(r["test_dates"], r["test_actual"],
                     label="Actual", marker="o", alpha=0.7)         
        axes[i].plot(r["test_dates"], r["best_forecast"],
                     label=f"{r['winner']} Forecast",
                     marker="x", linestyle="--")                    
        axes[i].set_title(
            f"{r['drug']} — Winner: {r['winner']} (RMSE: {r['best_rmse']})"
        )
        axes[i].set_xlabel("Date")
        axes[i].set_ylabel("Sales")
        axes[i].legend()
        axes[i].tick_params(axis="x", rotation=45)                

    plt.tight_layout()                                              

    save_path = os.path.join(BASE_DIR, "forecast_comparison.png")  
    plt.savefig(save_path, dpi=150)
    plt.close()                                                     
    print(f"Plot saved to {save_path}")