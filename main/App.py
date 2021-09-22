'''
Created on 14 Jul 2021

@author: matth
'''
import streamlit as st
from pages import Cleaner as cp  # Cleaning functionality page
from pages import ML_Models as ml  # ML functionality page
from pages import Predict as pr  # ML functionality page


def main():
    st.sidebar.image("media/Red_White_QUB_Logo.jpg", use_column_width=True)
    page = st.radio('', ('CSV Data Cleaner', 'Create ML Classifier', 'Make Predictions'))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    if page == 'CSV Data Cleaner':
        cp.CleanerPagePrintOut()
    elif page == 'Create ML Classifier':
        ml.MLPagePrintOut()
    else:
        pr.PredictPagePrintOut()
    

if __name__ == "__main__":
    main()
