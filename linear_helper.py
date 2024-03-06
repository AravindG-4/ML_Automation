from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint



def RidgeReg(X_train, y_train):

    ridge = Ridge()

    param_dist = {
        'alpha': uniform(loc=0, scale=2),  # Uniform distribution for regularization parameter
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    }

    random_search = RandomizedSearchCV(
        estimator=ridge,
        param_distributions=param_dist,
        n_iter=10,  # Number of random combinations to try
        cv=5,  # Number of cross-validation folds
        scoring='neg_mean_squared_error',  # Use negative mean squared error as the evaluation metric
        random_state=42
    )

    random_search.fit(X_train, y_train)

    best_params = random_search.best_params_

    best_score = random_search.best_score_

    ridge1 = Ridge(**best_params)

    ridge1.fit(X_train, y_train)

    return ridge1



def svr_model(x_train, y_train):

    svr_model = SVR()

    param_dist = {
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'C': uniform(loc=0, scale=10),  # Uniform distribution for regularization parameter
        'epsilon': uniform(loc=0, scale=0.1),  # Uniform distribution for epsilon parameter
        'gamma': ['scale', 'auto'],
    }

    random_search = RandomizedSearchCV(
        svr_model,
        param_distributions=param_dist,
        n_iter=10,  # Number of random combinations to try
        cv=5,  # Number of cross-validation folds
        scoring='r2',  # Use R-squared as the evaluation metric
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )

    random_search.fit(x_train, y_train)

    best_params = random_search.best_params_

    svr_model1 = SVR(**best_params)

    svr_model1.fit(x_train, y_train)

    return svr_model1


def random_forest_regression(x_train, y_train):

    rf_regressor = RandomForestRegressor()

    param_dist = {
        'n_estimators': randint(10, 100),  # Number of trees in the forest
        'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at each split
        'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum depth of the tree
        'min_samples_split': randint(2, 10),  # Minimum number of samples required to split an internal node
        'min_samples_leaf': randint(1, 10),  # Minimum number of samples required to be at a leaf node
        'bootstrap': [True, False],  # Whether bootstrap samples are used when building trees
        'random_state': [42]
    }

    random_search = RandomizedSearchCV(
        rf_regressor,
        param_distributions=param_dist,
        n_iter=10,  # Number of random combinations to try
        cv=5,  # Number of cross-validation folds
        scoring='r2',  # Use R-squared as the evaluation metric
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )

    random_search.fit(x_train, y_train)

    best_params = random_search.best_params_

    rf_regressor1 = RandomForestRegressor(**best_params)

    rf_regressor1.fit(x_train, y_train)

    return rf_regressor1


def decision_tree_regression(x_train, y_train):

    dt_regressor = DecisionTreeRegressor()

    param_dist = {
        'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum depth of the tree
        'min_samples_split': randint(2, 10),  # Minimum number of samples required to split an internal node
        'min_samples_leaf': randint(1, 10),  # Minimum number of samples required to be at a leaf node
        'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at each split
        'random_state': [42]
    }

    random_search = RandomizedSearchCV(
        dt_regressor,
        param_distributions=param_dist,
        n_iter=10,  # Number of random combinations to try
        cv=5,  # Number of cross-validation folds
        scoring='r2',  # Use R-squared as the evaluation metric
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )

    random_search.fit(x_train, y_train)

    best_params = random_search.best_params_

    dt_regressor1 = DecisionTreeRegressor(**best_params)

    dt_regressor1.fit(x_train, y_train)

    return dt_regressor1


def knn_regression(x_train, y_train):

    knn_regressor = KNeighborsRegressor()

    param_dist = {
        'n_neighbors': randint(1, 20),  # Number of neighbors to use
        'weights': ['uniform', 'distance'],  # Weight function used in prediction
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algorithm used to compute the nearest neighbors
        'p': [1, 2]  # Power parameter for the Minkowski metric
    }

    random_search = RandomizedSearchCV(
        knn_regressor,
        param_distributions=param_dist,
        n_iter=10,  # Number of random combinations to try
        cv=5,  # Number of cross-validation folds
        scoring='r2',  # Use R-squared as the evaluation metric
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )

    random_search.fit(x_train, y_train)

    best_params = random_search.best_params_

    knn_regressor1 = KNeighborsRegressor(**best_params)

    knn_regressor1.fit(x_train, y_train)

    return knn_regressor1

