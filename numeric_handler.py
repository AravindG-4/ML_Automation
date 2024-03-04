import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ------- Numeric Handler --------
def get_numeric_dataframe(df, preference, target, num_feats, req_num_feats = None):
    total_df = df
    preference = preference
    # req_features = req_features
    req_num_feats = req_num_feats
    target = target
    
    data = df[num_feats]
    
    req_features = df.columns
    # rows = df.shape[0]
    # columns = len(req_features)
    
    num_feats = num_feats
    cat_feats = [feature for feature in df.columns if feature not in num_feats]
    
    null_percent = get_null_percent(data, num_feats)
    
    data = fill_null(data, null_percent, num_feats)
    
    for feature in num_feats:
        winsorize(data, feature)
        
    return data
        
    # global scaler
    # scaler = StandardScaler()
    # scaler.fit(data)
    
    # data = standardize(data = data)

# -----------------------------------------------------------------------

# ------- Null percent common for all features --------

def get_null_percent(df, req_features):
    null_percent = {}

    features_with_na=[features for features in req_features if df[features].isnull().sum()>1]


    for feature in req_features:
        if feature in features_with_na:
            null_percent[feature] = np.round(df[feature].isnull().mean()*100, 4)
        else:
            null_percent[feature] = 0
    return null_percent

# ----------------------------------------------------------

# ------- Fill Null --------

def fill_null(df, null_percent, num_feats, req_features):
    more_null_features = []
    for feature in num_feats:
        if null_percent[feature] <= 35:
            df[feature].fillna(df[feature].median(), inplace=True)
        else:
            # req_features.append(feature + "nan")
            df[feature + "nan"] = np.where(df[feature].isnull(), 1, 0)
            more_null_features.append(feature)
    return df

# -----------------------------------------------------------------------

# ------- Winsorize --------

def winsorize(data, column_name, z_threshold=3):
    # Make a copy of the column
    column_copy = data[column_name].copy()
    
    column_copy = column_copy.astype(float)

    # Calculate the mean and standard deviation
    mean_val = np.mean(column_copy)
    std_dev = np.std(column_copy)

    # Calculate the Z-score for each data point
    z_scores = (column_copy - mean_val) / std_dev

    # Identify data points with Z-scores beyond the threshold
    outliers = np.abs(z_scores) > z_threshold

    # Replace outliers with values at the threshold
    column_copy[outliers] = np.sign(column_copy[outliers]) * z_threshold * std_dev + mean_val

    # Update the original DataFrame with the modified column
    data[column_name] = column_copy

# -----------------------------------------------------------------------


# ------- Standardize --------
# def standardize(df = None, input_array = None):
#     if df is not None:
#         return scaler.transform(df)
#     else:
#         return scaler.transform(input_array)
    