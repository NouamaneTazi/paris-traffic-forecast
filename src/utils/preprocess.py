from darts.timeseries import TimeSeries
from darts.utils.missing_values import fill_missing_values
from darts.dataprocessing.transformers import Scaler


def fill_ts(ts):
    # Fill missing values
    filled_ts = fill_missing_values(ts, -1.0)

    # add missing indicator component
    missing_indicator = ts.univariate_component(0)
    missing_indicator = TimeSeries.from_times_and_values(
        missing_indicator.time_index,
        missing_indicator.pd_dataframe().isna(),
        freq=missing_indicator.freq,
        columns=["missing"],
    )
    filled_ts = filled_ts.stack(missing_indicator)
    return filled_ts


def preprocess_data(all_arcs):
    for arc in all_arcs:
        # fill missing values
        arc["ts"] = fill_ts(arc["ts"])

        # separate past_covariates from targets
        target_noeuds = ["Lecourbe-Convention", "Convention-Blomet"]

        # training_cutoff = arc["ts"].time_index[-24 * 7]  # val = last week

        # set series to be predicted
        arc["targets"] = arc["ts"].univariate_component(0)
        arc["targets"] = arc["targets"].stack(arc["ts"].univariate_component(1))

        # train val split
        # train, val = arc["targets"].split_after(training_cutoff)

        # Normalize the time series (note: we avoid fitting the transformer on the validation set)
        transformer = Scaler()
        arc["train_scaled"] = transformer.fit_transform(arc["targets"])
        arc["targets_scaler"] = transformer
        # arc["val_scaled"] = transformer.transform(val)
        # arc["targets_scaled"] = transformer.transform(arc["targets"])

        # set past covariates
        arc["past_covs"] = arc["ts"].univariate_component(2)
        arc["past_covs"] = arc["past_covs"].stack(arc["ts"].univariate_component(3))
        # past_covs_train, past_covs_val = arc["past_covs"].split_after(training_cutoff)

        # transform past covs
        past_covs_scaler = Scaler()
        # past_covs_scaler.fit(past_covs_train)
        arc["past_covs_scaled"] = past_covs_scaler.fit_transform(arc["past_covs"])
        arc["past_covs_scaler"] = past_covs_scaler

    return all_arcs
