# This script is meant to import data I aggregated in Power BI and generate a forecast that is then saved into Excel. The forecasting library used is Prophet, made by Facebook for tasks like these.
def run_forecast():
    # Import libraries and tools
    import time
    import os
    import pandas as pd
    import warnings

    warnings.filterwarnings("ignore")
    import numpy as np
    from prophet import Prophet

    startTime = time.time()
    # Change working directory to where my live data is stored, makes it easier to import
    os.chdir("private data removed here")

    # Create a dataframe of the data for Python work
    df = pd.read_csv(
        "data.csv", parse_dates=["Date"], usecols=["private data removed here"],
    )
    df.rename(columns={"Date": "ds", "MonthSum": "y"}, inplace=True)
    df.y = df.y.astype("float")
    # Create new dataframe to save forecast results
    # global fcst_all
    fcst_all = pd.DataFrame()

    # Forecast function that loops through EVERY combo of cost center and cost element, generating a unique forecast for each one

    cost_centers = df["private data removed here"].unique()
    for center in cost_centers:
        cost_center_df = df.loc[df["private data removed here"] == center]
        if len(cost_center_df) < 12:
            pass
        else:
            cost_center_df = cost_center_df.drop(columns=["private data removed here"])
            cost_center_df.set_index("ds")
            my_model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=False,
                seasonality_prior_scale=1.0,
                yearly_seasonality=4,
                changepoint_prior_scale=1.0,
                seasonality_mode="multiplicative",
                interval_width=0.95,
            ).fit(cost_center_df)
            future_dates = my_model.make_future_dataframe(periods=3, freq="MS")
            forecast = my_model.predict(future_dates)
            forecast["private data removed here"] = center
            forecast["y"] = cost_center_df["y"].reset_index(drop=True)
            fcst_all = pd.concat((fcst_all, forecast))
        fcst_all = fcst_all[
            [
                "ds",
                "private data removed here",
                "y",
                "yhat",
                "yhat_lower",
                "yhat_upper",
            ]
        ]
        fcst_all = fcst_all.drop_duplicates()
        fcst_all.to_excel("ProphetForecast-new.xlsx", index=False)
    print("----------------------DONE----------------------")
    endTime = time.time()
    print("Took %s hours to calculate." % ((endTime - startTime) / 60 / 60))


if __name__ == "__main__":
    run_forecast()
