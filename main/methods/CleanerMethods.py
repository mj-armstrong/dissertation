'''
Created on 14 Jul 2021

@author: matth
'''
import pandas as pd
import base64
from future.builtins.misc import isinstance


def get_csv_df(file):
    '''
    Summary:
    Checks if file in the uploader is a CSV file.
    If it is it reads the CSV file to a pandas dataframe.
    
    Parameters:
    file - object in streamlit uploader.
    
    Returns:
    df - pandas dataframe.
    '''
    # get extension and read file
    extension = file.name.split('.')[-1].upper()
    if extension == 'CSV':
        df = pd.read_csv(file)
        
    return df


def checkListIsNum(colItems):
    '''
    Summary:
    Checks a list to see if they are all of type number OR None.
    
    Parameters:
    colItems - list, items in a pandas dataframe column.
    
    Returns:
    isNumList - boolean - True if all items in list are of type number or None, else returns False. 
    '''
    isNumList = True
    for item in colItems:
        if not (isinstance(item, int) or isinstance(item, float) or item == 'NaN'):
            isNumList = False
    return isNumList


def cleanDataFrame(df, ef_options):
    '''
    Summary:
    Performs cleaning functions on a pandas dataframe.
    Performs 'Remove row' function first.
    
    Parameters:
    df - pandas dataframe - dataset to be cleaned.
    ef_options - {index : [column header, option selected]}, datatype: {int : [str, str]}
    
    Returns:
    df - pandas dataframe - dataset that has had cleaning functions applied to it.
    '''
    for col in ef_options:
        if ef_options[col][1] == 'Remove row':
            df.dropna(subset=[ef_options[col][0]], inplace=True)
            
    for col in ef_options:
        if ef_options[col][1] == 'Do nothing':
            pass
        elif ef_options[col][1] == 'Mean':
            x = df[ef_options[col][0]].mean()
            df[ef_options[col][0]].fillna(x, inplace=True)
        elif ef_options[col][1] == 'Median':
            x = df[ef_options[col][0]].median()
            df[ef_options[col][0]].fillna(x, inplace=True)
        elif ef_options[col][1] == 'Mode':
            x = df[ef_options[col][0]].mode()[0]
            df[ef_options[col][0]].fillna(x, inplace=True)
    
    df = df.reset_index(drop=True)
              
    return df


def get_table_download_link(df, file_name):
    '''
    Note - Not oringally created by Matthew Armstrong, taken from anonymous author on forum.
    Edited to add coloured link and the ability to have custom file name.
    Also added '.csv' to file name to improve user experience upon downloading file.
    
    Summary:   
    Generates an HTML link to allow the data in a pandas dataframe to be downloaded.
    
    Parameters:
    df - dataframe to be downloaded as CSV file.
    
    Returns:
    href - string, HTML link.
    '''
    
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download =' + file_name + ' style="color: #96000d">Download file</a>'
    
    return href
    
