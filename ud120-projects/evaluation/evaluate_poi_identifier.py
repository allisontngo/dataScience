#!/usr/bin/python


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn.cross_validation import train_test_split

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

x_train, x_test, y_train, y_test = train_test_split(features, labels, \
									test_size=0.3, random_state=42)

### it's all yours from here forward!  
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
pred = clf.predict(x_test)
print 'accuracy: ', accuracy_score(y_test, pred)
print 'number of POI in test: ', pred.sum()
print 'number of people in test: ', len(pred)
print '0 poi accuracy: ', (len(pred)-pred.sum())/ len(pred)
print  confusion_matrix(y_test, pred)
print 'precision: ', precision_score(y_test, pred)
print 'recall: ', recall_score(y_test, pred)