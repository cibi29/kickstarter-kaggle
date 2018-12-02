# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 18:23:16 2018

@author: cibip
"""

import datetime as dt
import sys
import os
import os.path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier



# import matplotlib
import matplotlib.pyplot as plt


# pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.0f}'.format

'''
Abstract (

Introduction (Arnav)

Data Analysis (Arnav)

Implementation (Arnav, Tanmay, Cibi) => {Decision Trees/Random Forest, Logistic Regression}

Results (Arnav, Tanmay, Cibi)

Conclusion

Future Work

Related Work (Arnav, Tanmay, Cibi)

'''


def pre_processing(data):
    data.rename(columns=lambda x: x.strip(), inplace=True)
    
    data['deadline'] = pd.to_datetime(data['deadline'], errors='coerce')
    
    data['launched'] = pd.to_datetime(data['launched'], errors='coerce')
    
    data['goal'] = pd.to_numeric(data['goal'], errors='coerce')
    
    data['pledged'] = pd.to_numeric(data['pledged'], errors='coerce')
    
    data['usd pledged'] = pd.to_numeric(data['usd pledged'], errors='coerce')
    
    data['backers'] = pd.to_numeric(data['backers'], errors='coerce')
    
    data.dropna(inplace=True)

    print("these are the different column values")
    print(data.columns.values)
    data['Difference'] = (data['deadline'] - data['launched']).dt.days
    print(data.columns.values)
    print("the time durations of the projects are ")
    print(data['Difference'])


def visualization(data):
    def state_distribution():
        data['state'].value_counts().plot(kind='bar', color='#696969')
        plt.title('Kickstarter Project Status')
        plt.ylabel('# of projects')
        plt.xlabel('State')
        plt.show()

    def main_category_distribution():
        data['main_category'].value_counts().plot(kind='bar', color='#696969')
        plt.title('Kickstarter Main Category Distribution')
        plt.ylabel('# of projects')
        plt.xlabel('Main Categories')
        plt.show()

    def country_distribution():
        data['country'].value_counts().plot(kind='bar', color='#696969')
        plt.title('Kickstarter Country Distribution')
        plt.ylabel('# of Projects')
        plt.xlabel('Country')
        plt.show()

    def pledge_goal_ratio_state():
        ratio = data.groupby('state').agg({'pledged': np.mean, 'goal': np.mean})
        ratio['ratio'] = ratio['pledged'] / ratio['goal']
        ratio['ratio'].sort_values(ascending=False).plot(kind='bar', color='#696969')
        plt.title('Pledge to Goal Ratio per State')
        plt.ylabel('Pledge to Goal Ratio')
        plt.xlabel('State')
        plt.show()

    def pledge_goal_ratio_mc():
        ratio = data.groupby('main_category').agg({'pledged': np.mean, 'goal': np.mean})
        ratio['ratio'] = ratio['pledged'] / ratio['goal']
        ratio['ratio'].sort_values(ascending=False).plot(kind='bar', color='#696969')
        plt.title('Pledge to Goal Ratio per Main Category')
        plt.ylabel('Pledge to Goal Ratio')
        plt.xlabel('Main Category')
        plt.show()

    def project_outcome():
        table = data.pivot_table(index='main_category', columns='state', values='ID', aggfunc='count')
        # print(table)
        table['total'] = table.sum(axis=1)
        for column in table.columns[:5]:
            table[column] = table[column] / table['total']
        table.iloc[:, :5].plot(kind='bar', stacked=True, figsize=(9, 6),
                                  color=['#034752', '#88C543', 'black', '#2ADC75', 'white'])
        plt.title('Project Outcome by Category on Kickstarter')
        plt.legend(loc=2, prop={'size': 9})
        plt.xlabel('')
        plt.ylabel('Percentage of Projects')
        plt.show()

    # state_distribution()
    # main_category_distribution()
    # country_distribution()
    # pledge_goal_ratio_state()
    # pledge_goal_ratio_mc()
    project_outcome()

processed_cat = []

def label_creator(labels):
    global processed_cat
    encoder = LabelEncoder()
    encoder.fit(labels)
    processed_cat.append(encoder.transform(labels))
    return encoder.transform(labels)

def pre_processing_classification(data):
    # Use Label Encoder to encode string data.
    #excluding "useless" features
    global processed_cat
    
    data = data.drop(['ID','name','currency','launched','deadline','pledged'],axis=1)
    
    
    #creating labels for each of the column headers
    #string features- features to be converted from strings to integers 
    
    
    string_features = ['category','main_category','country','state']
    
    for feature in string_features:
        exec("label_creator(data."+feature+")")
    
    processed_cat = np.asarray(processed_cat)
    data = data.drop(string_features,axis=1)
    
    data = data.to_dict(orient='records')
    vec = DictVectorizer()
    numerical_features = vec.fit_transform(data).toarray()
    processed_cat = processed_cat.transpose()
    
    y = processed_cat[:,3]
    #concatenating both features
    cat_features = processed_cat[:,:3]
    X = np.concatenate((numerical_features,cat_features),axis = 1)  
    
    return X,y
    
    
    #remember that state is the last value sin 
    
    
    
    
    
    #Using a label encoder 
    
    
    
    
    pass


def main():
    csv_filepath = os.path.join('data', 'ks-projects-201801.csv')
    data = pd.read_csv(csv_filepath, usecols=range(13))
    
    pre_processing(data)
    visualization(data)
    X,y = pre_processing_classification(data)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state= 20)
    
    classifiers = ["RandomForestClassifier","DecisionTreeClassifier]
    
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    acc_test = clf.score(X_test, y_test)
    
    clf2 = DecisionTreeClassifier(random_state = 10)
    clf2.fit(X_train,y_train)
    acc_test2 = clf2.score(X_test,y_test)
    
    print ("Test Accuracy:", acc_test)
    print ("Test Accuracy:", acc_test2)

if __name__ == '__main__':
    main()
