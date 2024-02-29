import pandas as pd
import numpy as np
import streamlit as st



def get(dataframe,feature):
    global global_dataframe
    global global_features
    global_dataframe = dataframe.copy()
    global_features = feature.copy()


def drop_categorical_feature(dataframe,feature):

    no_of_unique  = len(dataframe[feature].unique())
    length_of_dataframe = len(dataframe)


    if no_of_unique > (0.7 * length_of_dataframe):
        global_dataframe[feature].drop(axis = 1 , inplace = True)
        st.write(f"Dropped feature {feature} from Dataframe")