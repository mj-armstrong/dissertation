'''
Created on 24 Aug 2021

@author: matth
'''

import streamlit as st
import seaborn as sns
import yellowbrick.classifier as ybc

    
def removeColumns(df, columnsIncluded):
    '''
    Summary:
    Removes columns from a dataframe based on user selection.
    
    Parameters:
    df - pandas dataframe, dataset to be edited.
    columnsIncluded - list, column of headers in list that are NOT going to be removed from dataset. 
    
    Returns:
    df - pandas dataframe, dataset with columns removed.
    '''
    for col in columnsIncluded:
        if not columnsIncluded[col]:
            df.drop(col, axis=1, inplace=True)
    return df

        
def checkColumnsAllFalse(columnsIncluded):
    '''
    Summary:
    Checks if no columns have been selected by the user.
    Error checking, as cannot proceed if dataset is empty.
    Prints message for user if they have not selected any columns.
    
    Parameters:
    columnsIncluded - list, column of headers in list that are NOT going to be removed from dataset. 
    
    Returns:
    allFalse - boolean, True if there are no columns in list, else returns False.    
    '''
    allFalse = True
    for col in columnsIncluded:
        if columnsIncluded[col]:
            allFalse = False
            
    if allFalse == True:
        st.write('No columns selected, cannot proceed.')
        st.write('Please select some columns.')
            
    return allFalse


def splitNumericCategoriesGetBounds(num_of_categories):
    '''
    Summary:
    Gathers the parameters that will be used to split a numeric column of a dataset into discrete categories.
    
    Parameters:
    num_of_categories - int, the number of categories that the user wants to split the column into.
    
    Returns:
    bounds - dictionary, {index : [category_name, upper_bound_value*]}, datatype {int : [string, number]}
    
    *note - last index will not have an upper_bound_value
    '''
    num_of_categories = int(num_of_categories)
    count = 1
    divider = st.selectbox('Upper Boundary Divider',
                           ['Less than (boundary values will go in higher value category)',
                            'Less than or Equal to (boundary values will go in lower value category)'])
    if divider == 'Less than (boundary values will go in higher value category)':
        divider = 0
    else:
        divider = 1
        
    while count <= num_of_categories:
        if count == num_of_categories:
            if count == 2:
                cat_2_name = st.text_input('2nd Category Name', value='Cat 2')
            elif count == 3:
                cat_3_name = st.text_input('3rd Category Name', value='Cat 3')
            else:
                cat_4_name = st.text_input('4th Category Name', value='Cat 4')
        elif count == 1:
            cat_1_name = st.text_input('1st Category Name', value='Cat 1')
            cat_1_ub = st.number_input('1st Category - Upper Bound')
        elif count == 2:
            cat_2_name = st.text_input('2nd Category Name', value='Cat 2')
            cat_2_ub = st.number_input('2nd Category - Upper Bound')
        else:
            cat_3_name = st.text_input('3rd Category Name', value='Cat 3')
            cat_3_ub = st.number_input('3rd Category - Upper Bound')
        
        count += 1
        
    if num_of_categories == 2:
        bounds = {0:divider, 1:[cat_1_name, cat_1_ub], 2:[cat_2_name]}
    elif num_of_categories == 3:
        bounds = {0:divider, 1:[cat_1_name, cat_1_ub], 2:[cat_2_name, cat_2_ub], 3:[cat_3_name]}
    else:
        bounds = {0:divider, 1:[cat_1_name, cat_1_ub], 2:[cat_2_name, cat_2_ub], 3:[cat_3_name, cat_3_ub], 4:[cat_4_name]}
    
    return bounds


def boudaryValuesValid(bounds, num_of_categories):
    '''
    Summary:
    Checks that the boundary values selected by the user are valid.
    Valid boundary values cannot be equal and boundary_value_0 < boundary_value_1 etc.
    
    Parameters:
    bounds - dictionary, {index : [category_name, upper_bound_value*]}, datatype {int : [string, number]}
    num_of_categories - int, the number of categories that the user wants to split the column into.
    
    Returns:
    valid - boolean, returns true if boundary values are valid, else returns false.
    '''
    valid = False
    num_of_categories = int(num_of_categories)
    
    if num_of_categories == 2:
        valid = True
    elif num_of_categories == 3:
        if bounds[1][1] < bounds[2][1]:
            valid = True
    else:
        if bounds[1][1] < bounds[2][1] and bounds[2][1] < bounds[3][1]:
            valid = True
    
    if valid == False:
        st.write('Invalid Boundary Values')
        st.write('Cat 1 Upper Bound < Cat 2 Upper Bound etc.')
    
    return valid


def boudaryNamesValid(bounds, num_of_categories):
    '''
    Summary:
    Checks that the boundary names selected by the user are valid.
    Valid boundary names cannot be the same.
    
    Parameters:
    bounds - dictionary, {index : [category_name, upper_bound_value*]}, datatype {int : [string, number]}
    num_of_categories - int, the number of categories that the user wants to split the column into.
    
    Returns:
    valid - boolean, returns true if boundary names are valid, else returns false.
    '''
    valid = True
    num_of_categories = int(num_of_categories)
    
    if num_of_categories == 2:
        if bounds[1][0] == bounds[2][0]:
            valid = False
    elif num_of_categories == 3:
        if (bounds[1][0] == bounds[2][0]) or (bounds[1][0] == bounds[3][0]) or (bounds[2][0] == bounds[3][0]):
            valid = False
    else:
        if (bounds[1][0] == bounds[2][0]) or (bounds[1][0] == bounds[3][0]) or (bounds[1][0] == bounds[4][0]) or (bounds[2][0] == bounds[3][0]) or (bounds[2][0] == bounds[4][0]) or (bounds[3][0] == bounds[4][0]):
            valid = False
    
    if valid == False:
        st.write('Invalid Category Names')
        st.write('Multiple categories cannot have the same name.')
    
    return valid


def splitNumericCategories0(bounds, num_of_categories, df, model_class):
    '''
    Summary:
    Alters a dataframe so that the class column is changed from numeric values to discrete categories as dictated by the user.
    Uses 'less than' to divide boundaries.
    
    Parameters:
    bounds - dictionary, {index : [category_name, upper_bound_value*]}, datatype {int : [string, number]}
    num_of_categories - int, the number of categories that the user wants to split the column into.
    df - pandas dataframe, the dataset that is to be altered.
    model_class - str, name of column header for class column selected by user.
    
    Returns:
    df - pandas dataframe, returns altered dataframe.
    '''
    num_of_categories = int(num_of_categories)
    df = df.astype({model_class: str})
    index = 0
    for item in df[model_class]:
        if item == 'nan':
            df.at[index, model_class] = None
        elif float(item) < bounds[1][1]:
            df.at[index, model_class] = bounds[1][0]
        elif num_of_categories == 2:
            df.at[index, model_class] = bounds[2][0]
        elif float(item) < bounds[2][1]:
            df.at[index, model_class] = bounds[2][0]
        elif num_of_categories == 3:
            df.at[index, model_class] = bounds[3][0]
        elif float(item) < bounds[3][1]:
            df.at[index, model_class] = bounds[3][0]
        else:
            df.at[index, model_class] = bounds[4][0]
        index += 1
    
    return df


def splitNumericCategories1(bounds, num_of_categories, df, model_class):
    '''
    Summary:
    Alters a dataframe so that the class column is changed from numeric values to discrete categories as dictated by the user.
    Uses 'less than or greater than' to divide boundaries.
    
    Parameters:
    bounds - dictionary, {index : [category_name, upper_bound_value*]}, datatype {int : [string, number]}
    num_of_categories - int, the number of categories that the user wants to split the column into.
    df - pandas dataframe, the dataset that is to be altered.
    model_class - str, name of column header for class column selected by user.
    
    Returns:
    df - pandas dataframe, returns altered dataframe.
    '''
    num_of_categories = int(num_of_categories)
    df = df.astype({model_class: str})
    index = 0
    for item in df[model_class]:
        if item == 'nan':
            df.at[index, model_class] = None
        elif float(item) <= bounds[1][1]:
            df.at[index, model_class] = bounds[1][0]
        elif num_of_categories == 2:
            df.at[index, model_class] = bounds[2][0]
        elif float(item) <= bounds[2][1]:
            df.at[index, model_class] = bounds[2][0]
        elif num_of_categories == 3:
            df.at[index, model_class] = bounds[3][0]
        elif float(item) <= bounds[3][1]:
            df.at[index, model_class] = bounds[3][0]
        else:
            df.at[index, model_class] = bounds[4][0]
        index += 1
    
    return df


def runClassifier(df, model_class, method, n_neighbour):
    '''
    Summary:
    Calls a method (selectClassifier) to create a scikit learn classifier using the supplied dataset and the users selected criteria.
    Creates train test split (using a 70/30 split) in the dataframe.
    Tests the Classifier.
    Prints classifier metrics to the frontend.
    
    Parameters:
    df - panda dataframe, dataframe on which the classifier is based.
    model_class - string, identifying the dataframe column that is the class.
    method - string, the machine learning algorithm that the user has selected.
    n_neighbour - int, the number of nearest neighbours the user has selected if using a nearest neighbour classifier.
    
    Returns:
    classifier - scikit learn classifier object.
    '''

    # Spliting dataset into training and test data
    from sklearn.model_selection import train_test_split
    X = df.drop(model_class, axis=1)
    y = df[model_class]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    
    classifier = selectClassifier(method, n_neighbour)
    
    # create classifier
    classifier.fit(X_train, y_train)
    # test classifier
    predictions = classifier.predict(X_test)
    
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_test, predictions)
    dl = list(set(df[model_class]))
    dl = sorted(dl)
    
    # Classifier Metrics
    st.subheader('Classifier Metrics')
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    accuracy = accuracy_score(y_test, predictions)
    st.write("Classifier Accuracy = " + str(round(accuracy, 2)))
    f1 = f1_score(y_test, predictions, average='weighted')
    st.write("Weighted Average F1 Score = " + str(round(f1, 2)))
    precision = precision_score(y_test, predictions, average='weighted')
    st.write("Weighted Average Precision = " + str(round(precision, 2)))
    recall = recall_score(y_test, predictions, average='weighted')
    st.write("Weighted Average Recall = " + str(round(recall, 2)))
    
    st.subheader('Pairplot')
    fig = sns.pairplot(df, hue=model_class, palette='Set1')
    st.pyplot(fig)
    
    # Confusion Matrix
    confusion_matrix = ConfusionMatrixDisplay(cm, display_labels=dl)
    confusion_matrix.plot(cmap='Reds')
    confusion_matrix.ax_.set(
                title='Confusion Matrix',
                xlabel='Predicted Value',
                ylabel='Actual Value')
    st.pyplot()
    
    ybc.classification_report(classifier, X_train, y_train, X_test, y_test, classes=dl, support=True, cmap='Reds')
    st.pyplot()
    
    return classifier

    
def selectClassifier(method, n_neighbour):
    '''
    Summary:
    Creates the classifier from the user selection.
    
    Parameters:
    method - string, classifier selcted by the user.
    n_neighbour - int, the number of nearest neighbours the user has selected if using a nearest neighbour classifier.
    
    Returns:
    classifier - scikit learn classifier object.
    '''
    
    if method == 'Decision Tree':
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier()
    elif method == 'Random Forest':
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier()
    elif method == 'Gaussian Naive Bayes':
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
    elif method == 'n Nearest Neighbours':
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors=n_neighbour)
    elif method == 'Zero R':
        from sklearn.dummy import DummyClassifier
        classifier = DummyClassifier(strategy="most_frequent")
    elif method == 'Ada Boost':
        from sklearn.ensemble import AdaBoostClassifier
        classifier = AdaBoostClassifier()
    elif method == 'Quadratic Discriminant Analysis':
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        classifier = QuadraticDiscriminantAnalysis()
    
    return classifier


def checkDatasetClassNotEmpty(classColumn):
    '''
    Summary:
    Error Checking.
    Checks to see if any value in the class column is empty.
    If there are any empty cells the classifier cannot be created from the dataset.
    If false prints message to screen.
        
    Parameters:
    classColumn - list, all the items in the datasets class column.
    
    Returns:
    check - boolean, returns False if any items in the list are of type None. Else returns True.
    '''
    check = True
    
    for item in classColumn:
        if item == None:
            check = False
            
    if check == False:
        st.write('At least one of the items in the class column is empty.')
        st.write('This is not appropriate for buiding a classifier, please fix this dataset.')
        st.write('This can be done using the CSV Data Cleaner (see top of page).')
    
    return check
    
