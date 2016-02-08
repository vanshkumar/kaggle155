import sklearn 
import numpy as np
import os
import itertools

# Takes in estimator, parameters to search over, train data X, train data y,
# test data X, test data y
def grid_search(estim, params, x_dev, y_dev, x_val, y_val):
    keys = params.keys()

    vals = params.values()
    all_combos = [s for s in itertools.product(*vals)]

    best_combo = ''
    best_score = 0.0

    for combo in all_combos:
        for j in range(len(keys)):
            setattr(estim, keys[j], combo[j])

        estim.fit(x_dev, y_dev)

        score = estim.score(x_val, y_val)

        # print score

        if score > best_score:
            best_combo = combo
            best_score = score

    # Set params of best estimator
    for j in range(len(keys)):
        setattr(estim, keys[j], best_combo[j])

    print best_combo
    print estim.get_params()

    return estim, best_score