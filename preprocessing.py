import streamlit as st
from handle_categorical_features import *






def set_numerical_features(dataframe , features):

    global global_numerical_features
    global_numerical_features = [feature for feature in features if len(dataframe[feature]) <  50] 






def set_categorical_features(dataframe , features):

    global global_categorical_features
    global_categorical_features = [feature for feature in features if (len(dataframe[feature]) >  50)]



def remove_columns(dataframe,features):
     
    for feature in features:

        if dataframe[feature].isnull().sum() >= (0.5 * len(dataframe)):

            features.remove(feature)

    global_dataframe = dataframe[features]
    

def remove_features_with_priority(global_dataframe , global_features , number):

    null_amount = dict()

    for feature in global_features:

        null_amount[feature] = global_dataframe[feature].isnull().sum()
    
    sorted_dict_by_values = dict(sorted(null_amount.items(), key=lambda item: item[1]))
    global_dataframe = global_dataframe[list(sorted_dict_by_values.keys())[:number]]


    


def set_global_processing_dataframe(dataframe,features):

    global global_dataframe 
    global  global_features
    global_dataframe = dataframe.copy()
    global_features = features.copy()
    set_categorical_features(global_dataframe , global_features)
    set_numerical_features(global_dataframe , global_features)
    st.write("Set DataFrame",global_dataframe)

def update_categorical_dataframe(dataframe):
    global_dataframe = dataframe


def start_preprocessing(dataframe,features,number,preferrence,target):
   
   set_global_processing_dataframe(dataframe , features)


   if preferrence == 'Default':
       print(dataframe)
       remove_columns(global_dataframe , global_categorical_features) 


   elif preferrence == 'Select Own features':
       pass
   elif preferrence == 'Selct Number of Features': 
       
       remove_features_with_priority(global_dataframe,global_features,number)

   
   categoric_handler(dataframe,preferrence,target,features)
   update_categorical_dataframe(get_categorical_dataframe())
   
   
   st.write(global_dataframe)