import numpy as np
import pandas as pd

import torch

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.utils.timeseries_generation import datetime_attribute_timeseries
import warnings

warnings.filterwarnings("ignore")


import logging

logging.disable(logging.CRITICAL)

import argparse
import os
import json

# sagemaker
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current_host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])

    parser.add_argument("--train_fileName", type=str, default="convention-2014-2021.csv-proc.csv")
    parser.add_argument("--epochs", type=int, default=1)

    args = parser.parse_args()

    # ## load data
    # df = pd.read_csv("processed_data/convention-2014-2021.csv-proc.csv").set_index('time')
    df = pd.read_csv(args.data_dir + "/" + args.train_fileName).set_index("time")
    df.index = pd.to_datetime(df.index, utc=False)
    df.sort_index(inplace=True)
    df

    value_cols = ["debit", "occupation"]
    cov_cols = ["etat_barre"]
    ts = TimeSeries.from_dataframe(df, value_cols=value_cols, fill_missing_dates=True, freq="H")
    past_covs = TimeSeries.from_dataframe(df, value_cols=cov_cols, fill_missing_dates=True, freq="H")
    series = ts

    # ## model

    # ts.drop_before(pd.Timestamp("2021-11-20")).univariate_component(0).plot(label='Débit horaire', new_plot=True)

    # Create training and validation sets:
    # training_cutoff = pd.Timestamp('2021-12-01')
    # train, val = ts.split_after(training_cutoff)
    train = ts
    past_covs_train = past_covs

    # Normalize the time series (note: we avoid fitting the transformer on the validation set)
    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)
    # val_transformed = transformer.transform(val)
    series_transformed = transformer.transform(ts)

    # create year, month and integer index covariate series
    covariates = datetime_attribute_timeseries(series, attribute="year", one_hot=False)
    covariates = covariates.stack(datetime_attribute_timeseries(series, attribute="month", one_hot=False))
    covariates = covariates.stack(datetime_attribute_timeseries(series, attribute="day", one_hot=False))
    covariates = covariates.stack(datetime_attribute_timeseries(series, attribute="hour", one_hot=False))
    covariates = covariates.stack(
        TimeSeries.from_times_and_values(
            times=series.time_index, values=np.arange(len(series)), columns=["linear_increase"]
        )
    )
    covariates

    # transform past covs
    past_scaler_covs = Scaler()
    past_scaler_covs.fit(past_covs_train)
    past_covariates_transformed = past_scaler_covs.transform(past_covs_train)

    # transform covariates (note: we fit the transformer on train split and can then transform the entire covariates series)
    scaler_covs = Scaler()
    cov_train = covariates
    scaler_covs.fit(cov_train)
    covariates_transformed = scaler_covs.transform(covariates)

    # default quantiles for QuantileRegression

    input_chunk_length = 168  # week
    forecast_horizon = 24  # day
    my_model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=forecast_horizon,
        hidden_size=64,
        lstm_layers=1,
        num_attention_heads=4,
        dropout=0.3,
        batch_size=128,
        n_epochs=args.epochs,
        add_relative_index=False,
        add_cyclic_encoder="day",
        likelihood=None,  # QuantileRegression is set per default
        loss_fn=torch.nn.MSELoss(),
        random_state=42,
    )

    print("Start fitting model")
    my_model.fit(
        train_transformed,
        past_covariates=past_covariates_transformed,
        future_covariates=covariates_transformed,
        verbose=True,
    )

    print(f"Saving model to {args.model_dir}")
    torch.save(my_model, args.model_dir + "/model.pth")

    past_covariates_transformed = past_covariates_transformed.append_values([1] * 24 * 10)
    pred_ts = my_model.predict(n=24 * 5, num_samples=1, past_covariates=past_covariates_transformed)
    pred_ts = transformer.inverse_transform(pred_ts)

    pred_ts.to_csv(args.model_dir + "preds.csv")
    print("Preds saved successfully!")
