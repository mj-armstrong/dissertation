'''
Created on 28 Jul 2021

@author: matth

This module contains the method that is responsible for the front end of the 1st tab of the application. 
'''

import streamlit as st
import pandas as pd
from methods import CleanerMethods as cl  # custom script


def CleanerPagePrintOut():
    '''
    Summary:
    Provides frontend elements of 'CSV Data Cleaner' tab using the streamlit framework.
    '''
    st.title('CSV Cleaner')
    dataset = st.file_uploader("Upload CSV File", type=['csv'])
    if dataset is not None:
        st.write('Original Dataset')
        df = pd.read_csv(dataset, na_values="NaN")
        st.write(df)
        headers = list(df.columns)
        
        with st.sidebar:
            with st.form('treatment_form'):
                st.header('Empty Cell Treatment:')
                st.write('Notes')
                st.write('- Remove row function will always be performed first.')
                st.write('- Mode will insert the lowest/first alphabetically mode if there is more than one.')
                ef_options = {}
                index = 0
                for col in headers:
                    ef_options[index] = [col]
                    if cl.checkListIsNum(df[col]):
                        ef_options[index].append(st.selectbox(
                        str(index) + ') ' + str(col), [
                        'Do Nothing',
                        'Remove row',
                        'Mean',
                        'Mode',
                        'Median']))
                    else:
                        ef_options[index].append(st.selectbox(
                        str(index) + ') ' + str(col), [
                        'Do Nothing',
                        'Remove row',
                        'Mode']))
                    
                    index += 1
                
                submitted = st.form_submit_button("Submit")
                
        if submitted:
            st.write('Revised Dataset - saved to cache')
            revised_df = cl.cleanDataFrame(df, ef_options)
            st.write(revised_df)
            st.header('Download revised dataset as CSV file')
            # Add revised csv to name
            file_name = dataset.name[0:-4] + '_revised.csv'
            # Remove white space to ensure link works, replace with _
            file_name = file_name.replace(" ", "_")
            st.markdown(cl.get_table_download_link(revised_df, file_name), unsafe_allow_html=True)
            st.session_state.revised_df = revised_df
    
