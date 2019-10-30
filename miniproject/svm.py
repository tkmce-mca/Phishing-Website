# -*- coding: utf-8 -*-

#importing libraries
from sklearn.externals import joblib
import inputScript


#input url
print("enter url")
url = input()

#load the pickle file
classifier = joblib.load('final_models/svm_final.pkl')

#checking and predicting
checkprediction = inputScript.main(url)
prediction = classifier.predict(checkprediction)
print(prediction)
