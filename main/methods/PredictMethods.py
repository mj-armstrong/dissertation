'''
Created on 6 Sep 2021

@author: matth
'''

def checkHeaders(headers, classifier_headers):
    check = False
    if(all(item in headers for item in classifier_headers)):
        check = True
        
    return check

def updateDataSet(df, classifier_headers):
    df = df[classifier_headers]
    return df