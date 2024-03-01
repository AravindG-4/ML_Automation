import streamlit as st





def start_preprocessing(dataframe,features,number):
    set_global(dataframe,features)
    if number is not None:
        st.write(number)
        st.write(dataframe)
    else:
        st.write("Select the number of features...")
