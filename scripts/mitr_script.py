import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

# =======================================================================================
# Settings
random_state = 42
test_size = 0.3
grid_search_spaces = {
    "LinearRegression": {"fit_intercept": [True, False]},
    "Lasso": {"alpha": np.linspace(0.000001, 1, 200)},
    "DecisionTreeRegressor": {
        "max_depth": [None, 5, 10, 25, 50],
        "max_features": [None, "sqrt", "log2", 0.2, 0.4, 0.6, 0.8, 1],
        "min_samples_leaf": [1, 2, 4, 6, 8, 10],
        "min_samples_split": [2, 4, 6, 8, 10],
    },
    "RandomForestRegressor": {
        "n_estimators": [50, 100, 150],
        "criterion": ["squared_error", "absolute_error"],
        "min_samples_split": [2, 4],
        "max_features": [None, "sqrt", "log2", 1],
    },
    "KNeighborsRegressor": {
        "n_neighbors": [1, 2, 4, 6, 8, 10, 14, 17, 20],
        "weights": ["uniform", "distance"],
        "leaf_size": [1, 5, 10, 15, 20, 25, 30],
    },
}
np.random.seed(random_state)

# =======================================================================================
# Read data
X = pd.read_csv(
    "https://raw.githubusercontent.com/aims-umich/pyMAISE/develop/pyMAISE/data/crx.csv"
)
print(X)
Y = pd.read_csv(
    "https://raw.githubusercontent.com/aims-umich/pyMAISE/develop/pyMAISE/data/powery.csv"
)
print(Y)

# =======================================================================================
# Train/test split
xtrain, xtest, ytrain, ytest = train_test_split(
    X, Y, test_size=0.3, random_state=random_state
)

# =======================================================================================
# Scale data using min-max scaling
xscaler = MinMaxScaler()
yscaler = MinMaxScaler()
xtrain = xscaler.fit_transform(xtrain)
xtest = xscaler.transform(xtest)
ytrain = yscaler.fit_transform(ytrain)
ytest = yscaler.transform(ytest)

print(f"xtrain shape: {xtrain.shape}")
print(f"ytrain shape: {ytrain.shape}")
print(f"xtest shape: {xtest.shape}")
print(f"ytest shape: {ytest.shape}")

# =======================================================================================
# Grid search
best_params = {}
best_models = {}
for key, value in grid_search_spaces.items():
    model = None
    if key == "RandomForestRegressor":
        model = RandomForestRegressor(random_state=random_state)
    elif key == "DecisionTreeRegressor":
        model = DecisionTreeRegressor(random_state=random_state)
    else:
        model = eval(key + "()")
    print(f"GridSearchCV: {key}")
    search = GridSearchCV(
        model,
        param_grid=value,
        cv=ShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state),
    )
    search.fit(xtrain, ytrain)

    # Get best model and parameters
    best_params[key] = search.best_params_
    best_models[key] = search.best_estimator_


# =======================================================================================
# Scoring
def score_model(scores, models, x, y, split):
    for key, value in models.items():
        y_hat = value.predict(x)
        scores[f"{split} MAE"].append(mean_absolute_error(y, y_hat))
        scores[f"{split} MSE"].append(mean_squared_error(y, y_hat))
        scores[f"{split} RMSE"].append(np.sqrt(mean_squared_error(y, y_hat)))
        scores[f"{split} R2"].append(r2_score(y, y_hat))
    return scores


scores = {
    "Model Types": ["Linear", "Lasso", "DT", "RF", "KN"],
    "Train MAE": [],
    "Train MSE": [],
    "Train RMSE": [],
    "Train R2": [],
    "Test MAE": [],
    "Test MSE": [],
    "Test RMSE": [],
    "Test R2": [],
}

scores = score_model(scores, best_models, xtrain, ytrain, "Train")
scores = score_model(scores, best_models, xtest, ytest, "Test")

scores = pd.DataFrame(scores)
print(scores)
scores.to_csv("mitr_testing_metrics.csv", index=False)
