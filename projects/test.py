from ray import tune
from ray.air import session


# def objective(config):  # ①
#     score = config["a"] ** 2 + config["b"]
#     return {"score": score}


# search_space = {  # ②
#     "a": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
#     "b": tune.choice([1, 2, 3]),
# }
# tuner = tune.Tuner(objective, param_space=search_space)

# results = tuner.fit()
# print(results.get_best_result(metric="score", mode="min").config)


def objective(x, a, b):
    return a * (x**0.5) + b


def trainable(config: dict):
    intermediate_score = 0
    for x in range(20):
        intermediate_score = objective(x, config["a"], config["b"])
        session.report({"score": intermediate_score})  # This sends the score to Tune.


tuner = tune.Tuner(trainable, param_space={"a": 2, "b": 4})
results = tuner.fit()
