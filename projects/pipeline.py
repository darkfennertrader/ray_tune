import os
import random
from pprint import pprint
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

blocks = max(int(os.cpu_count()) - 2, 1)
blocks = 2

BASE_DIR1 = "data/AUD-CAD"
BASE_DIR2 = "data/AUD-CHF"
BASE_DIR3 = "data/AUD-JPY"
NR_ASSETS = 2
FILENAME = "overall"
NUM_TRIALS = 50
TICKS_YEAR = 32  # minutes in 252 trading days


ray.init(ignore_reinit_error=True)
print()
print("*" * 100)
pprint(ray.cluster_resources(), indent=2)
print("*" * 100)
print()
# print(os.cpu_count())


########################     TRAINABLE     ###############################
def _open_csv(filepath: str) -> pd.DataFrame:
    return pd.read_csv(
        filepath,
        parse_dates=["Datetime"],
        index_col="Datetime",
        date_parser=lambda x: pd.to_datetime(x, utc=True),  # type: ignore
    )


########################     TRANFORM DATA     #################################
# # remove random samples from original file


def _save_to_csv(pathfile: str, data: pd.DataFrame) -> None:
    data.to_csv(pathfile)


# # read original data
# input_df = _open_csv(BASE_DIR + FILENAME + ".csv")


# #########       Function to transform a pandas dataframe     ########
def transform_batch(
    data: pd.DataFrame,
    # base_dir: str,
    percent_to_remove: float = 0.2,
) -> pd.DataFrame:
    # print(data.head())
    # normalize value
    data["Close"] = data["Close"] / data["Close"].iloc[0]
    print(data.head(2))
    print(f"batch shape: {data.shape}")
    # store original index
    original_idx = data.index
    # remove random data
    idx_list = random.sample(
        range(0, data.shape[0]),
        int(data.shape[0] * percent_to_remove),
    )
    print(f"removed {len(idx_list)} samples")

    data_dropped = data.drop(data.index[idx_list])
    print(f"data_dropped shape: {data_dropped.shape}")
    data_dropped.reset_index(inplace=True)
    data_dropped = data_dropped.reindex(original_idx)
    print(data_dropped.head())

    # # _save_to_csv(base_dir + FILENAME + "_sampled.csv", data)
    # # _save_to_csv(base_dir + "missing_sampled.csv", data_dropped)

    # return data_dropped.to_frame()
    print()
    return data


#############################     PIPELINE     ################################

main_list = [
    f"{BASE_DIR1}/overall.csv",
    f"{BASE_DIR2}/overall.csv",
    f"{BASE_DIR3}/overall.csv",
]

main_array = np.array(main_list)

chunked_array = np.array_split(main_list, 5)
splitted_list = [list(array) for array in chunked_array]
# remove empty lists
splitted_list = [x for x in splitted_list if x]

print(splitted_list)

###############################################################################

print("\n(1) CREATE DATASET:")
dataset = ray.data.read_csv(
    [
        f"{BASE_DIR1}/overall.csv",
        f"{BASE_DIR2}/overall.csv",
        f"{BASE_DIR3}/overall.csv",
    ],
)


print("\n(2) TRANSFORM DATASET:")
dataset = dataset.drop_columns(["Open", "High", "Low"])
# print(dataset.show(1))
dataset.count()
print(dataset)
# print(dataset.default_batch_format())

# dataset = dataset.split(2)
# repartition does not distinguish between different assets
# dataset = dataset.repartition(2)
# # print(np.round(dataset.materialize().size_bytes() / (1024) ** 2, 3))

transformed_dataset = dataset.map_batches(transform_batch)
transformed_dataset.show(1)
# print(transformed_dataset)

# print("\nCONSUME THE DATASET:")


# @ray.remote
# class Trainer:
#     def __init__(self, config: dict, index):
#         self.config = config
#         self.index = index
#         # print(f"\nCONFIG: {config}")

#     def __repr__(self):
#         return f"MyActor(index={self.index})"

#     def train(self, shard: ray.data.Dataset) -> int:
#         for batch in shard.iter_batches(batch_size=TICKS_YEAR, batch_format="pandas"):
#             print("Batch shape:")
#             print(batch)
#             print(batch.shape)
#         return shard.count()


# trainers = [Trainer.remote(i) for i in range(NR_ASSETS)]

# shards = transformed_dataset.split(n=len(trainers), equal=True, locality_hints=trainers)
# print(shards)

# result = ray.get([w.train.remote(s) for w, s in zip(trainers, shards)])
# print(result)

# # Delete trainer actor handle references, which should terminate the actors.
# del trainers
