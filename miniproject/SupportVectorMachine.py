# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.externals import joblib

def build():
   
    dataset = pd.read_csv("datasets/phishcoop.csv")
    dataset = dataset.drop('id', 1) 
    x = dataset.iloc[: , :-1].values
    y = dataset.iloc[:, -1:].values

   
    from sklearn.cross_validation import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state =0 )

     
    from sklearn.model_selection import GridSearchCV
    parameters = [{'C':[1, 10, 100, 1000], 'gamma': [ 0.1, 0.2,0.3, 0.5]}]
    grid_search = GridSearchCV(SVC(kernel='rbf' ),  parameters,cv =5, n_jobs= -1)
    grid_search.fit(x_train, y_train)

    
    print("Best Accurancy =" +str( grid_search.best_score_))
    print("best parameters =" + str(grid_search.best_params_)) 
def train():
    

    classifier = SVC(C=1000, kernel = 'rbf', gamma = 0.2 , random_state = 0)
    classifier.fit(x_train, y_train)
def test():
    #predicting the tests set result
    y_pred = classifier.predict(x_test)

   joblib.dump(classifier, 'final_models/svm_final.pkl')
