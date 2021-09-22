'''
Created on 14 Jul 2021

@author: matth
'''
import pandas as pd
import base64
from future.builtins import isinstance


def get_csv_df(file):
    # get extension and read file
    extension = file.name.split('.')[-1].upper()
    if extension == 'CSV':
        df = pd.read_csv(file)
        
    return df


def checkListIsNum(colItems):
    isNumList = True
    for item in colItems:
        if not (isinstance(item, int) or isinstance(item, float) or item == 'NaN'):
            isNumList = False
    return isNumList


def cleanDataFrame(df, ef_options):
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
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download =' + file_name + ' style="color: #96000d">Download file</a>'
    
    return href
    
