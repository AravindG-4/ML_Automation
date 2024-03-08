import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
import streamlit as st
import pickle
from imblearn.over_sampling import SMOTEN
from sklearn.preprocessing import StandardScaler
from linear_models import *
from classify_models import *
from DynoML import number

global scaler 
scaler = StandardScaler()

def click_button():
    st.session_state.clicked = True


import pandas as pd
from sklearn.utils import resample

def up_down_sample_data(dataframe, target):
    """
    Up-sample or down-sample the DataFrame based on the target variable.

    Parameters:
        dataframe (DataFrame): The DataFrame containing the data.
        target (str): The name of the target variable column.

    Returns:
        DataFrame: The up-sampled or down-sampled DataFrame.
    """
    # Check if the number of unique values in the target variable is less than or equal to 50
    if dataframe[target].nunique() <= 50:
        target_counts = dataframe[target].value_counts().to_dict()
        
        # Determine the minority and majority classes
        minority_class = min(target_counts, key=target_counts.get)
        majority_class = max(target_counts, key=target_counts.get)
        
        # Calculate required_count dynamically
        total_samples = len(dataframe)
        num_classes = len(target_counts)
        avg_class_samples = total_samples / num_classes
        
        # Calculate the required count based on the average number of samples per class
        required_count = int(avg_class_samples)
        
        # Check if upsampling or downsampling is required
        if target_counts[minority_class] < required_count or target_counts[majority_class] > required_count:
            # Upsample minority class and downsample majority class
            minority_data = dataframe[dataframe[target] == minority_class]
            majority_data = dataframe[dataframe[target] == majority_class]
            
            # Upsample minority class
            minority_upsampled = resample(minority_data,
                                          replace=True,
                                          n_samples=required_count,
                                          random_state=42)  # You can choose any random_state
            
            # Downsample majority class
            majority_downsampled = resample(majority_data,
                                            replace=False,
                                            n_samples=required_count,
                                            random_state=42)  # You can choose any random_state
            
            # Concatenate the upsampled minority and downsampled majority
            sampled_dataframe = pd.concat([minority_upsampled, majority_downsampled])
            
            st.write("Resampled dataframe")
            return sampled_dataframe
        else:
            return dataframe
    else:
        # If the number of unique values in the target variable is more than 50, return the original DataFrame
        return dataframe



def feature_selection(x_train, y_train, num_features):
    lasso_cv = LassoCV(cv=5)
    lasso_cv.fit(x_train, y_train)

    # Identify features with non-zero coefficients
    model = SelectFromModel(lasso_cv, max_features=num_features, threshold=-np.inf)
    model.fit(x_train, y_train)

    # Get the indices of selected features
    selected_feature_indices = model.get_support(indices=True)

    # Get the names of selected features
    selected_feature_names = x_train.columns[selected_feature_indices].tolist()

    return selected_feature_names


def start_training(dataframe,target):
        
        st.write('In train models')

        dataframe.drop_duplicates(keep='first', inplace=True)
        # Assuming X contains features and y contains target values
        st.write("In train models")


        dataframe = up_down_sample_data(dataframe, target)

        X = dataframe.drop(columns=[target] , axis = 1)
        y = dataframe[target]

        y_unique = len(y.unique())

        


        

        # Split the data into training and testing sets (e.g., 80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        selected_features = feature_selection(X_train, y_train, number)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

        st.write("Final Dataframe", X_train)

        X_train_scaled = scaler.fit_transform(X_train)
        # Transform test data
        X_test_scaled = scaler.transform(X_test)
        
        


        dict = None

            

        if y_unique >= 50 :

            st.write("X_train",X_train_scaled)
            dict = linear_models(X_train_scaled, y_train, X_test_scaled, y_test)
                            
                            
        else:
            
            st.write("X_train",X_train_scaled)
            dict = classification_models(X_train_scaled, y_train, X_test_scaled, y_test)


        if dict is not None:
            st.divider()  # 👈 Draws a horizontal rule
            st.header('Download Models')
                                    
            for model , path in dict.items():     
                    
                    st.write(model , path)
                    st.write(f'***{model}***')
                    
                    st.download_button(f'Download',file_name= model+'.pkl' , data = open(path, 'rb').read())      

