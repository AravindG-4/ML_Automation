import streamlit as st
from handle_categorical_features import *
from numeric_handler import *
from train_models import *






def set_numerical_features(dataframe , features):

    global global_numerical_features
    global_numerical_features = [feature for feature in features if len(dataframe[feature].unique()) >  50] 






def set_categorical_features(dataframe , features):

    global global_categorical_features
    global_categorical_features = [feature for feature in features if (len(dataframe[feature].unique())) < 50]



def remove_columns(dataframe,features):
     
    for feature in features:
        features_to_remove = []
        if dataframe[feature].isnull().sum() >= (0.5 * len(dataframe)):
         features = [feature for feature in features if feature not in features_to_remove]
    
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
    st.write(global_categorical_features)
    st.write(global_numerical_features)







def start_preprocessing(dataframe,features,number,preferrence,target):
   
 with st.spinner("Preprocessing the data..."):
    set_global_processing_dataframe(dataframe , features)


    if preferrence == 'Default':
        st.write("In Default")
        remove_columns(global_dataframe , global_categorical_features) 


    elif preferrence == 'Select Own features':
        st.write("In Select own features")
        pass
    elif preferrence == 'Selct Number of Features': 
        st.write("In Select no of features")

        remove_features_with_priority(global_dataframe,global_features,number)
    
    
    categoric_handler(dataframe,preferrence,target,global_categorical_features)
    categoric_dataframe = get_categorical_dataframe()
    numeric_dataframe = get_numeric_dataframe(dataframe, preferrence, target, global_numerical_features , number)

    final_dataframe = pd.concat([categoric_dataframe , numeric_dataframe],axis = 1)
    st.success("Preprocessing completed ", icon="✅")
    st.write(final_dataframe)
    return final_dataframe
