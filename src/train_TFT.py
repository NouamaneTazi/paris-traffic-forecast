import numpy as np
import pandas as pd
import joblib

# from matplotlib import pyplot as plt

import torch

from darts.models import TFTModel
import warnings

from utils import preprocess_data

warnings.filterwarnings("ignore")


import logging

logging.disable(logging.CRITICAL)

import argparse
import os
import json

# # set environment for sagemaker fir local testing
# os.environ["SM_HOSTS"] = '["to-be-filled-by-sagemaker"]'
# os.environ["SM_CURRENT_HOST"] = "to-be-filled-by-sagemaker"
# os.environ["SM_MODEL_DIR"] = "models"
# os.environ["SM_CHANNEL_TRAINING"] = "tmp"


# sagemaker
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current_host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data_dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])

    parser.add_argument("--train_fileName", type=str, default="voi-convention2021.pkl")  # ALL
    parser.add_argument("--pretrain_epochs", type=int, default=1)
    parser.add_argument("--finetune_epochs", type=int, default=1)

    args = parser.parse_args()

    target_noeuds = ["Lecourbe-Convention", "Convention-Blomet"]

    # ## load data
    if args.train_fileName == "ALL":
        data_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith(".pkl")]
        all_arcs = []
        for f in data_files:
            all_arcs.extend(joblib.load(f))
    else:
        all_arcs = joblib.load(f"{args.data_dir}/{args.train_fileName}")  # voi-convention1.pkl
    print(f"Loaded {len(all_arcs)} arcs:")
    for arc in all_arcs:
        print(f"{arc['noeud_amont']} {arc['noeud_aval']}")

    # ## preprocess data
    all_arcs = preprocess_data(all_arcs)

    for arc in all_arcs:
        # set final target arc
        if [arc["noeud_amont"], arc["noeud_aval"]] == target_noeuds:
            target_arc = arc

    # ## train model
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
        add_relative_index=False,
        add_cyclic_encoder="day",
        likelihood=None,  # QuantileRegression is set per default
        loss_fn=torch.nn.MSELoss(),
        random_state=42,
    )

    try:
        print("Fitting model on all data...")
        my_model.fit(
            [arc["train_scaled"] for arc in all_arcs],
            past_covariates=[arc["past_covs_scaled"].univariate_component(0) for arc in all_arcs],
            future_covariates=[arc["past_covs_scaled"].univariate_component(1) for arc in all_arcs],
            verbose=True,
            epochs=args.pretrain_epochs,
        )

        print(f"Saving pretrained model to {args.model_dir}")
        torch.save(my_model, args.model_dir + "/TFT-model-pretrained.pth")

        print("Finetuning model on target data...")
        my_model.fit(
            target_arc["train_scaled"],
            past_covariates=target_arc["past_covs_scaled"].univariate_component(0),
            future_covariates=target_arc["past_covs_scaled"].univariate_component(1),
            verbose=True,
            epochs=args.finetune_epochs,
        )

        print(f"Saving model to {args.model_dir}")
        torch.save(my_model, args.model_dir + "/TFT-model.pth")

        # ## predict on test data
        forecast_horizon = 24 * 5  # predict 5 days
        past_covs = target_arc["past_covs_scaled"].univariate_component(0).append_values(np.ones(forecast_horizon))
        future_covs = target_arc["past_covs_scaled"].univariate_component(1).append_values(np.zeros(forecast_horizon))

        # NOTE: fitting on multiple series gives the "need past_covariates" error
        # to avoid that, we fit on a single series one last time and then predict using only future_covariates
        pred_ts = my_model.predict(
            forecast_horizon,
            series=target_arc["train_scaled"],
            past_covariates=past_covs,
            future_covariates=future_covs,
        )
        # pred_ts.univariate_component(0).plot()
        # target_arc["val_scaled"].univariate_component(0).plot()
        # plt.show()

        # pred_ts = target_arc["scaler"].inverse_transform(pred_ts)
        pred_ts.to_csv(args.model_dir + "/preds.csv")
        print("Preds saved successfully!")

    except:
        print("An error happened")
        print(f"Saving model to {args.model_dir}")
        torch.save(my_model, args.model_dir + "/TFT-model.pth")
