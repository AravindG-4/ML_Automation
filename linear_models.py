import numpy as np
import pandas as pd
from sklearn import metrics
from linear_helper import * 
import streamlit as st
import pickle

def linear_models(x_train, y_train, x_test, y_test):

    with st.spinner("Training models..."):
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
        




    # Combine the data into a list of tuples
        data = list(zip(linear_result['Models'],
                    linear_result['R2 Score'],
                    linear_result['Mean Absolute Error'],
                    linear_result['Mean Squared Error'],
                    linear_result['Root Mean Squared Error']))

    # Sort the list of tuples based on 'R2 Score'
        sorted_data = sorted(data, key=lambda x: x[1], reverse=True)

    # Convert the sorted list of tuples back to a dictionary
        sorted_linear_result = {
        'Models': [item[0] for item in sorted_data],
        'R2 Score': [item[1] for item in sorted_data],
        'Mean Absolute Error': [item[2] for item in sorted_data],
        'Mean Squared Error': [item[3] for item in sorted_data],
        'Root Mean Squared Error': [item[4] for item in sorted_data]
    }






        
        linear_result_df = pd.DataFrame(linear_result, index = None)
        
        st.write(linear_result_df)
        
        list_of_models = [RidgeRegression,DecisionTreeRegressor,RandomForestRegressor,SupportVectorRegressor,KNeighborsRegressor]

        #Pickle files

        linear_pickles = {}
        

        pickle.dump(RidgeRegression, open('pickles/RidgeRegression.pkl', 'wb'))
        pickle.dump(DecisionTreeRegressor, open('pickles/DecisionTreeRegression.pkl', 'wb'))
        pickle.dump(RandomForestRegressor, open('pickles/RandomForestRegression.pkl', 'wb'))
        pickle.dump(SupportVectorRegressor, open('pickles/SupportVectorRegression.pkl', 'wb'))
        pickle.dump(KNeighborsRegressor, open('pickles/KNeighborsRegression.pkl', 'wb'))
        
        linear_pickles['RidgeRegression'] = 'pickles/RidgeRegression.pkl'
        linear_pickles['DecisionTreeRegressor'] = 'pickles/DecisionTreeRegression.pkl'
        linear_pickles['RandomForestRegressor'] = 'pickles/RandomForestRegression.pkl'
        linear_pickles['SupportVectorRegressor'] = 'pickles/SupportVectorRegression.pkl'
        linear_pickles['KNeighborsRegressor'] = 'pickles/KNeighborsRegression.pkl'
        
        st.success('Training completed!', icon="✅")
        st.toast('Ready!', icon = "🥞")
        return linear_pickles




