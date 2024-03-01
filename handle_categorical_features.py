import pandas as pd
import numpy as np
import streamlit as st
from categorical_helper_functions import *

def set_global(dataframe , features):
    global global_dataframe
    global global_features
    global global_categorical_features
    global global_numerical_features
    global_categorical_features = [ feature for feature in features if len(global_dataframe[feature]) <= 25]
    global_numerical_features = [ feature for feature in features if len(global_dataframe[feature]) >= 25]
    global_dataframe = dataframe
    global_features = features


def handle_caterical_feature(dataframe,features):
    set_global(dataframe,features)
    global_dataframe = fill_categorical_na(global_dataframe,global_categorical_features)
    global_dataframe = encode_features(features)

def get_global():
    return global_dataframe