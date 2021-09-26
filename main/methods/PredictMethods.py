'''
Created on 6 Sep 2021

@author: matth
'''


def checkHeaders(headers, classifier_headers):
    '''
    Summary:
    Error Checking
    Checks to see that all columns required by the classifier are in the dataset uploaded by the user.
    
    Parameters:
    headers - list, all headers in the dataset that has been uploaded for classification.
    classifier_headers - list, all headers in the dataset that was used to build the classifier.
    
    Returns:
    check - boolean, returns True if the dataset is valid.
    '''
    check = False
    if(all(item in headers for item in classifier_headers)):
        check = True
        
    return check


def updateDataSet(df, classifier_headers):
    '''
    Summary:
    Selects columns from the uploaded dataset that are required for use in the classifier.
    
    Parameters:
    df - pandas dataframe, dataset that has been uploaded by user for classification.
    classifier_headers - pandas dataframe, columns that are required for use in the classifier model.
    
    Returns:
    df - pandas dataframe, dataset with only required columns included.
    '''
    df = df[classifier_headers]
    return df
