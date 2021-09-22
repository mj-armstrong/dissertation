'''
Created on 29 Jul 2021

@author: matth
'''

import streamlit as st
import pandas as pd
import seaborn as sns
from methods import ML_ModelsMethods as mm  # custom script
from methods import CleanerMethods as cl


def MLPagePrintOut():
    
    df_set = False
    spliting = False
    categories_valid = True
    bounds = {}
    
    st.title('Create Machine Learning Classifier')
    
    if 'revised_df' not in st.session_state:
        st.write('No cached dataset, please upload file.')
        dataset = st.file_uploader("Upload CSV File", type=['csv'])
        if dataset is not None:
            df = pd.read_csv(dataset, na_values="NaN")
            df_set = True   
    else:
        option = st.selectbox(
                        'Select Dataset', [
                        'Cached Dataset',
                        'Upload New Dataset'])
        if option == 'Upload New Dataset':
            dataset = st.file_uploader("Upload CSV File", type=['csv'])
            if dataset is not None:
                df = pd.read_csv(dataset, na_values="NaN")
                df_set = True
        else:
            df = st.session_state.revised_df
            df_set = True
            
    if df_set == True:
        st.write('Dataframe:')
        st.write(df)
        headers = list(df.columns)
        
        try:
            st.write('Correlation Heatmap')
            df_corr = df.corr()
            sns.heatmap(df_corr, annot=True, cmap='Reds')
            st.pyplot()
        except ValueError:
            st.write('Dataset is not appropriate for Correlation Heatmap')
        
        method = st.selectbox(
                                'Select ML Method', [
                                'Decision Tree',
                                'Random Forest',
                                'Gaussian Naive Bayes',
                                'n Nearest Neighbours',
                                'Ada Boost',
                                'Quadratic Discriminant Analysis',
                                'Zero R'])
        
        n_neighbour = -1
        if method == 'n Nearest Neighbours':
            n_neighbour = st.selectbox('Select n', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        model_class = st.selectbox('Select Class', headers)
        numeric_class = cl.checkListIsNum(df[model_class])
        
        if numeric_class:
                st.write('Split numeric class into discrete categories')
                num_of_categories = st.selectbox('Number of Categories', ['Don\'t Split', '2', '3', '4'], index=1)
                
        with st.form('treatment_form'):
            
            save_classifier = st.text_input(label='Classifier Name - Enter Name to Save Classifier')
            
            if numeric_class:
                if num_of_categories == '2' or num_of_categories == '3' or num_of_categories == '4':
                    spliting = True
                    bounds = mm.splitNumericCategoriesGetBounds(num_of_categories)
            
            st.write("Columns to be included in model:")
            
            column_included = {}
            
            for col in headers:
                if col != model_class:
                    column_included[col] = st.checkbox(col, value="True")
        
            submitted = st.form_submit_button("Submit")
        
        if submitted:
            
            model_df = mm.removeColumns(df, column_included)
                        
            if numeric_class and spliting:
                categories_valid = mm.boudaryValuesValid(bounds, num_of_categories) and mm.boudaryNamesValid(bounds, num_of_categories)
            
            if (not categories_valid) or mm.checkColumnsAllFalse(column_included):
                pass  # not placeholder
            else:
                if spliting:
                    if bounds[0] == 0:
                        model_df = mm.splitNumericCategories0(bounds, num_of_categories, model_df, model_class)
                    else:
                        model_df = mm.splitNumericCategories1(bounds, num_of_categories, model_df, model_class)
                
                st.write('Dataset used for model:')
                st.write(model_df)
                st.markdown(cl.get_table_download_link(model_df, 'model_dataset.csv'), unsafe_allow_html=True)
                
                if submitted:
                    
                    if mm.checkDatasetClassNotEmpty(model_df[model_class]):
                    
                        classifier = mm.runClassifier(model_df, model_class, method, n_neighbour)
                        
                        if save_classifier != '':
                            classifier_name = save_classifier + '_name'
                            classifier_class = save_classifier + '_class'
                            classifier_model = save_classifier + '_model'
                            classifier_headers = save_classifier + '_headers'
                            classifier_datatypes = save_classifier + '_datatypes'
                            st.session_state[classifier_name] = save_classifier
                            st.session_state[classifier_class] = model_class
                            st.session_state[classifier_model] = classifier
                            st.session_state[classifier_headers] = list(model_df.columns)
                            st.session_state[classifier_datatypes] = model_df.dtypes
        
