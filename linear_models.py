import numpy as np
import pandas as pd
from sklearn import metrics
from linear_helper import * 
import streamlit as st
import pickle

def linear_models(x_train, y_train, x_test, y_test):
    RidgeRegression = RidgeReg(x_train,y_train)
    DecisionTreeRegressor = decision_tree_regression(x_train,y_train)
    RandomForestRegressor = random_forest_regression(x_train, y_train)
    SupportVectorRegressor = svr_model(x_train, y_train)
    KNeighborsRegressor = knn_regression(x_train, y_train)
    
    linear_result = {'Models' : [], 'R2 Score': [], 'Mean Absolute Error': [], 'Mean Squared Error': [], 'Root Mean Squared Error': []}
    
    y_pred = RidgeRegression.predict(x_test)
    linear_result['Models'].append("RidgeRegression")
    linear_result['R2 Score'].append(metrics.r2_score(y_test, y_pred))
    linear_result['Mean Absolute Error'].append(metrics.mean_absolute_error(y_test, y_pred))
    linear_result['Mean Squared Error'].append(metrics.mean_squared_error(y_test, y_pred))
    linear_result['Root Mean Squared Error'].append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    
    y_pred = DecisionTreeRegressor.predict(x_test)
    linear_result['Models'].append('DecisionTreeRegressor')
    linear_result['R2 Score'].append(metrics.r2_score(y_test, y_pred))
    linear_result['Mean Absolute Error'].append(metrics.mean_absolute_error(y_test, y_pred))
    linear_result['Mean Squared Error'].append(metrics.mean_squared_error(y_test, y_pred))
    linear_result['Root Mean Squared Error'].append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    y_pred = RandomForestRegressor.predict(x_test)
    linear_result['Models'].append('RandomForestRegressor')
    linear_result['R2 Score'].append(metrics.r2_score(y_test, y_pred))
    linear_result['Mean Absolute Error'].append(metrics.mean_absolute_error(y_test, y_pred))
    linear_result['Mean Squared Error'].append(metrics.mean_squared_error(y_test, y_pred))
    linear_result['Root Mean Squared Error'].append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    y_pred = SupportVectorRegressor.predict(x_test)
    linear_result['Models'].append('SupportVectorRegressor')
    linear_result['R2 Score'].append(metrics.r2_score(y_test, y_pred))
    linear_result['Mean Absolute Error'].append(metrics.mean_absolute_error(y_test, y_pred))
    linear_result['Mean Squared Error'].append(metrics.mean_squared_error(y_test, y_pred))
    linear_result['Root Mean Squared Error'].append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    y_pred = KNeighborsRegressor.predict(x_test)
    linear_result['Models'].append('KNeighborsRegressor')
    linear_result['R2 Score'].append(metrics.r2_score(y_test, y_pred))
    linear_result['Mean Absolute Error'].append(metrics.mean_absolute_error(y_test, y_pred))
    linear_result['Mean Squared Error'].append(metrics.mean_squared_error(y_test, y_pred))
    linear_result['Root Mean Squared Error'].append(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    
    
    #Converting the DataFrame
    
    linear_result = dict(sorted(linear_result.items(), key=lambda item: item[1]))
    
    linear_result_df = pd.DataFrame(linear_result, index = None)
    
    st.write(linear_result_df)
    
    #Pickle files
    linear_pickles = {}
    
    pickle.dump(RidgeRegression, open('pickles\RidgeRegression.pkl', 'wb'))
    pickle.dump(DecisionTreeRegressor, open('pickles\DecisionTreeRegression.pkl', 'wb'))
    pickle.dump(RandomForestRegressor, open('pickles\RandomForestRegression.pkl', 'wb'))
    pickle.dump(SupportVectorRegressor, open('pickles\SupportVectorRegressor.pkl', 'wb'))
    pickle.dump(KNeighborsRegressor, open('pickles\KNeighborsRegressor.pkl', 'wb'))
    
    linear_pickles['RidgeRegression'] = 'pickles\RidgeRegression.pkl'
    linear_pickles['DecisionTreeRegressor'] = 'pickles\DecisionTreeRegressor.pkl'
    linear_pickles['RandomForestRegressor'] = 'pickles\RandomForestRegressor.pkl'
    linear_pickles['SupportVectorRegressor'] = 'pickles\SupportVectorRegressor.pkl'
    linear_pickles['KNeighborsRegressor'] = 'pickles\KNeighborsRegressor.pkl'
    
    return linear_pickles