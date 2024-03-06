from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

def logreg(x_train, y_train):
    logreg_model = LogisticRegression()


    param_dist = {
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'C': uniform(loc=0, scale=4),  # Uniform distribution for regularization parameter
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        'max_iter': [100, 300, 500, 1000],
        'random_state': [42]
    }

    # Create RandomizedSearchCV object
    random_search = RandomizedSearchCV(
        logreg_model,
        param_distributions=param_dist,
        n_iter=10,  # Number of random combinations to try
        cv=5,  # Number of cross-validation folds
        scoring='accuracy',  # Use accuracy as the evaluation metric
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )

    # Fit the RandomizedSearchCV object on the training data
    random_search.fit(x_train, y_train)

    # Get the best parameters
    best_params = random_search.best_params_

    # Create a new Logistic Regression model with the best parameters
    logreg_model = LogisticRegression(**best_params)

    # Fit the model on the entire training data
    logreg_model.fit(x_train, y_train)

    # Return the trained model
    return logreg_model


def decisionTree(x_train, y_train):
    dt_model = DecisionTreeClassifier()

    # Define the hyperparameter grid for RandomizedSearchCV
    param_dist_dt = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None] + list(range(5, 50)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'random_state': [42]
    }

    # Create RandomizedSearchCV object for Decision Tree
    random_search_dt = RandomizedSearchCV(
        dt_model,
        param_distributions=param_dist_dt,
        n_iter=10,
        cv=5,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )

    # Fit the RandomizedSearchCV object on the training data
    random_search_dt.fit(x_train, y_train)

    # Get the best parameters
    best_params_dt = random_search_dt.best_params_

    # Create a new Decision Tree model with the best parameters
    dt_model = DecisionTreeClassifier(**best_params_dt)

    # Fit the model on the entire training data
    dt_model.fit(x_train, y_train)

    # Return the trained Decision Tree model
    return dt_model


def randomForest(x_train, y_train):
    rf_model = RandomForestClassifier()

    # Define the hyperparameter grid for RandomizedSearchCV
    param_dist_rf = {
        'n_estimators': randint(10, 200),
        'criterion': ['gini', 'entropy'],
        'max_depth': [None] + list(range(5, 50)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 20),
        'bootstrap': [True, False],
        'random_state': [42]
    }

    # Create RandomizedSearchCV object for Random Forest
    random_search_rf = RandomizedSearchCV(
        rf_model,
        param_distributions=param_dist_rf,
        n_iter=10,
        cv=5,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )

    # Fit the RandomizedSearchCV object on the training data
    random_search_rf.fit(x_train, y_train)

    # Get the best parameters
    best_params_rf = random_search_rf.best_params_

    # Create a new Random Forest model with the best parameters
    rf_model = RandomForestClassifier(**best_params_rf)

    # Fit the model on the entire training data
    rf_model.fit(x_train, y_train)

    # Return the trained Random Forest model
    return rf_model


def gradientBoost(x_train, y_train):
    gb_model = GradientBoostingClassifier()

    # Define the hyperparameter grid for RandomizedSearchCV
    param_dist_gb = {
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
        'n_estimators': randint(50, 200),
        'max_depth': [3, 4, 5, 6, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0],
        'random_state': [42]
    }

    # Create RandomizedSearchCV object
    random_search_gb = RandomizedSearchCV(
        gb_model,
        param_distributions=param_dist_gb,
        n_iter=10,
        cv=5,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )

    # Fit the RandomizedSearchCV object on the training data
    random_search_gb.fit(x_train, y_train)

    # Get the best parameters
    best_params_gb = random_search_gb.best_params_

    # Create a new GradientBoostingClassifier model with the best parameters
    gb_model = GradientBoostingClassifier(**best_params_gb)

    # Fit the model on the entire training data
    gb_model.fit(x_train, y_train)

    # Return the trained model
    return gb_model


def adaBoost(x_train, y_train):
    ada_model = AdaBoostClassifier()

    # Define the hyperparameter grid for RandomizedSearchCV
    param_dist_ada = {
        'n_estimators': randint(50, 200),
        'learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3],
        'algorithm': ['SAMME', 'SAMME.R'],
        'random_state': [42]
    }

    # Create RandomizedSearchCV object
    random_search_ada = RandomizedSearchCV(
        ada_model,
        param_distributions=param_dist_ada,
        n_iter=10,
        cv=5,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )

    # Fit the RandomizedSearchCV object on the training data
    random_search_ada.fit(x_train, y_train)

    # Get the best parameters
    best_params_ada = random_search_ada.best_params_

    # Create a new AdaBoostClassifier model with the best parameters
    ada_model = AdaBoostClassifier(**best_params_ada)

    # Fit the model on the entire training data
    ada_model.fit(x_train, y_train)

    # Return the trained model
    return ada_model


def knn(x_train, y_train):
    knn_model = KNeighborsClassifier()

    # Define the hyperparameter grid for RandomizedSearchCV
    param_dist = {
        'n_neighbors': randint(1, 20),  # Random integer values for the number of neighbors
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2],  # 1 for Manhattan distance, 2 for Euclidean distance
    }

    # Create RandomizedSearchCV object
    random_search = RandomizedSearchCV(
        knn_model,
        param_distributions=param_dist,
        n_iter=10,  # Number of random combinations to try
        cv=5,  # Number of cross-validation folds
        scoring='accuracy',  # Use accuracy as the evaluation metric
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )

    # Fit the RandomizedSearchCV object on the training data
    random_search.fit(x_train, y_train)

    # Get the best parameters
    best_params = random_search.best_params_

    # Create a new KNN model with the best parameters
    best_knn_model = KNeighborsClassifier(**best_params)

    # Fit the model on the entire training data
    best_knn_model.fit(x_train, y_train)

    # Return the trained model
    return best_knn_model