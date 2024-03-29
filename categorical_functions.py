import pandas as pd
import numpy as np
import streamlit as st
import handle_categorical_features 

#*********************************************************************************************************************

# To set the DataFrame as Global variable, So it can be accessed from anywhere of the file



#*********************************************************************************************************************


#*********************************************************************************************************************

# To get the DataFrame from another file


    

#*********************************************************************************************************************

# To fill the categorical features with low number of missing values

def fill_categorical_null(dataframe,feature):
    st.write(f"In fill categorical null {feature}")

    if dataframe[feature].isnull().sum() < (0.35 * len(dataframe)):
        
        
        dataframe[feature].fillna(value = dataframe[feature].mode()[0],inplace = True)
        handle_categorical_features.update_categorical_dataframe(dataframe)    
        st.write((dataframe[feature]))
        

    else:

        fill_feature_with_more_null(dataframe , feature)




#*********************************************************************************************************************

# To encode the classes of a feature

def encode_features(dataframe,feature):
   
    a = dataframe[feature].unique()
    b = range(len(a))
    c = zip(a,b)
    d = dict(tuple(c))

    dataframe[feature] = dataframe[feature].map(d)

    handle_categorical_features.update_categorical_dataframe(dataframe)

#*********************************************************************************************************************8


# To get the classes of a feature with High frequency 

def get_classes_with_high_frequency(dataframe,feature):

    class_counts = dataframe[feature].value_counts()
    class_to_count = dataframe[feature].unique()
    class_with_high_frequency = []



    for i in class_to_count:
        
        class_frequency = class_counts.get(i, 0)
        l = len(dataframe)
        
        if class_frequency > (0.005 * l):
            
            class_with_high_frequency.append(i)
    return class_with_high_frequency


#*********************************************************************************************************************8


def create_rare_class(dataframe , feature):


    classes_with_high_frequency = get_classes_with_high_frequency(dataframe , feature)

    if classes_with_high_frequency:
        c = tuple(zip(classes_with_high_frequency,classes_with_high_frequency))
        d = dict(c)
        print(d)
        dataframe[feature] = dataframe[feature].map(d)
        dataframe[feature].fillna(value = 'Rare', inplace = True)

    handle_categorical_features.update_categorical_dataframe(dataframe)    
#*********************************************************************************************************************8


def fill_feature_with_more_null(dataframe,feature):

    dataframe[feature].fillna(value = 'Missing')
    handle_categorical_features.update_categorical_dataframe(dataframe)
    print(f"Filled null values of feature {feature} with class 'Missing'")


