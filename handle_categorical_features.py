import pandas as pd
import numpy as np
import streamlit as st
from categorical_functions import *

#******************************************************************************************************

def update_dataframe(dataframe):

    global_dataframe = dataframe


def update_features(features):

    global_features = features


def get_categorical_dataframe():
    return global_dataframe

def get_categorical_features():
    return global_features


def set_global(dataframe , features):
    global global_dataframe
    global global_features
    global_dataframe = dataframe
    global_features = features

#******************************************************************************************************
    
def categoric_handler(dataframe,preferrence,target,features):

    set_global(dataframe , features)


    for feature in global_features:

        fill_categorical_null(global_dataframe,feature)
        create_rare_class(global_dataframe , feature)
        encode_features(global_dataframe , feature)