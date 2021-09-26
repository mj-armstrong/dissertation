'''
Created on 14 Jul 2021

@author: matth


'''
import streamlit as st
from pages import Cleaner as cp  # Cleaning functionality page
from pages import ML_Models as ml  # ML functionality page
from pages import Predict as pr  # ML functionality page


def main():
    '''
    Summary:
    main method, initialises App.py.
    
    Provides elements of page that are consistent across tabs (sidebar image and radio menu).
    
    Runs methods for each tab based on the users menu selection.
    '''
    st.sidebar.image("https://raw.githubusercontent.com/mj-armstrong/dissertation/main/main/media/Red_White_QUB_Logo.jpg", use_column_width=True)
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
