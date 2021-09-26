'''
Created on 4 Sep 2021

@author: matth
'''

import streamlit as st
import pandas as pd
from future.builtins.misc import isinstance
from methods import PredictMethods as pm  # custom script
from methods import CleanerMethods as cl  # custom script


def PredictPagePrintOut():
    '''
    Summary:
    Provides frontend elements of 'Make Predictions' tab using the streamlit framework.
    '''
    
    st.title('Make Predictions')
    
    classifier_names = []
    
    if st.session_state != None:
        for item in st.session_state:
            if isinstance(item, str):
                if item[-4:] == 'name':
                    classifier_names.append(item[:-5])
                
                
                
    if classifier_names == []:
        st.write('No Classifiers Created')
    else:
                
        classifier_select = st.radio('', classifier_names)
        classifier_class = st.session_state[classifier_select + '_class']
        classifier_model = st.session_state[classifier_select + '_model']
        classifier_headers = st.session_state[classifier_select + '_headers']
        classifier_datatypes = st.session_state[classifier_select + '_datatypes']
        
        # Removing class from classifier_headers list
        try:
            classifier_headers.remove(classifier_class)
        except ValueError:
            pass
        
        dataset = st.file_uploader("Upload CSV File", type=['csv'])
        if dataset is not None:
            st.write('Dataset to be Classified')
            df = pd.read_csv(dataset, na_values="NaN")
            st.write(df)
            headers = list(df.columns)
            
            if classifier_class in headers:
                df = df.drop(classifier_class, axis=1)
                st.write('Note - Column to be classified has been overwritten with predictions.')
                
            try: 
                if not pm.checkHeaders(headers, classifier_headers):
                    st.write('Uploaded dataset does not include all columns used in this classifier.')
                    st.write('The dataset should include the following columns:')
                    for col in classifier_headers:
                        st.write(col)
                    st.write('Please revise the dataset and try again.')
                else:
                    classifier_dataset = pm.updateDataSet(df, classifier_headers)
                    
                    preds = classifier_model.predict(classifier_dataset)
                    
                    preds_df = pd.DataFrame(preds, columns=[classifier_class])
                    
                    predictions_dataset = pd.concat([df, preds_df], axis=1)
                    
                    st.write(predictions_dataset)
                    
                    st.markdown(cl.get_table_download_link(predictions_dataset, 'predictions_dataset.csv'), unsafe_allow_html=True)
            
            except ValueError:
                st.write('Datatype for columns in uploaded dataset does not match datatype used for the selected classifier.')
                st.write('Classifier datatypes:')
                index = 0
                for item in classifier_headers:
                    st.write(item + ' - datatype : ' + str(classifier_datatypes[index]))    
                    index += 1
