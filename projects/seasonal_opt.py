#   $ pip install py-spy
#   $ sudo chown root:root `which py-spy`
#   $ sudo chmod u+s `which py-spy`


import random
from time import time
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
import ray
from ray.air import session
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers.pb2 import PB2
from ray.tune.tune_config import TuneConfig

BASE_DIR = "./data/AUD-CAD/"
FILENAME = "overall"
NUM_TRIALS = 50
NR_SAMPLES = 3000000
random.seed(42)


# ray.init(num_cpus=16, ignore_reinit_error=True, object_store_memory=11**10)
ray.init()


def _log_transform(data: pd.DataFrame):
    return np.log(data + 1)


def _inverse_log(data: pd.DataFrame):
    return np.exp(data) - 1


def _open_csv(filepath: str) -> pd.DataFrame:
    return pd.read_csv(
        filepath,
        parse_dates=["Datetime"],
        index_col="Datetime",
        date_parser=lambda x: pd.to_datetime(x, utc=True),  # type: ignore
    )


def _save_to_csv(pathfile: str, data: pd.DataFrame) -> None:
    data.to_csv(pathfile)


# objective function to minimize
def objective(
    original_data: pd.Series, missing_data: pd.Series, period: int, seasonal: int
) -> float:
    missing_data = _log_transform(missing_data).copy()

    res = STL(
        missing_data.interpolate(method="linear"), period=period, seasonal=seasonal
    ).fit()
    # extract seasonal component
    seasonal_component = res.seasonal
    # De-seasonlise original data
    data_deseasonalised = missing_data - seasonal_component
    # Perform linear interpolation on de-seasonalised data
    data_deseasonalised_imputed = data_deseasonalised.interpolate(method="linear")
    # Add seasonal component back to get the final imputed time series
    df_imputed = data_deseasonalised_imputed + seasonal_component
    df_imputed = df_imputed.to_frame().rename(columns={0: "Close"})
    df_imputed = _inverse_log(df_imputed).to_numpy()

    original_data = original_data.to_numpy().reshape(-1, 1)

    return sum(((original_data - df_imputed) * 10) ** 2)[0]


###############################################################################
# remove random samples from original file
data = _open_csv(BASE_DIR + FILENAME + ".csv")
data = data["Close"][:NR_SAMPLES].copy()

# normalize value
data = data / data.iloc[0]

print(data.head())

print(data.shape)
original_idx = data.index

_save_to_csv(BASE_DIR + FILENAME + "_sampled.csv", data)

PERCENT_TO_REMOVE = 0.2
idx_list = random.sample(
    range(0, data.shape[0]),
    int(data.shape[0] * PERCENT_TO_REMOVE),
)

# print(sorted(idx_list))
print(f"removed {len(idx_list)} samples")

data_dropped = data.drop(data.index[idx_list]).iloc[:NR_SAMPLES]
print(data_dropped.shape)
data_dropped.to_frame().reset_index(inplace=True)
data_dropped = data_dropped.reindex(original_idx)
data_dropped = data_dropped / data_dropped.iloc[0]
_save_to_csv(BASE_DIR + "missing_sampled.csv", data_dropped)


##### with Ray Tune ###########################

start = time()


def trainable(config: dict, data: pd.DataFrame, data_dropped: pd.DataFrame):
    period, seasonal = config["period"], config["seasonal"]
    for _ in range(1):
        score = objective(data, data_dropped, period, seasonal)
        session.report({"score": score})


search_space = {
    "period": tune.randint(10, 200),
    "seasonal": 7,
}

pb2_scheduler = PB2(
    time_attr="time_total_s",
    metric="score",
    mode="min",
    perturbation_interval=600,
    hyperparam_bounds={"period": [10, 200], "seasonal": [7, 7]},
)


tuner = tune.Tuner(
    # use tune.with_parameters() to pass large quantity of data
    tune.with_parameters(trainable, data=data, data_dropped=data_dropped),
    tune_config=TuneConfig(
        num_samples=NUM_TRIALS,
    ),
    param_space=search_space,
)
results = tuner.fit()
print(results.get_best_result(metric="score", mode="min").config)

print(f"Ray Tune time elapsed: {(time()-start):.3f}")


###################################################################
# normal loop
# start = time()
# for period in range(6, 16):
#     res = objective(data, data_dropped, period, 7)
#     print(f"\nperiod: {period}, objective:{res:.6f}")

# print(f"Normal loop time elapsed: {(time()-start):.3f}")
