import pandas as pd
import numpy as np
import streamlit as st






def fill_categorical_na(dataframe,features):
    df = dataframe.copy()
    for feature in features:
        df[feature].fillna(value = df[feature].mode(),inplace = True)
        st.write(dataframe)
    return df

def encode_features(dataframe,features):
    df = dataframe.copy()
    for feature in features:
        a = len(df)
        b = range(a)
        c = zip(a,b)
        d = dict(tuple(c))
        df[feature].map(d)
    return df