import streamlit as st

st.session_state.stage = 0


if st.session_state.stage == 0:
    st.button('STart pre')
    st.session_state.stage = 1
    st.write('Hi')


if st.session_state.stage >= 1:
    name = st.button('start train')
    st.write('Helo')
    st.session_state.stage = 0
