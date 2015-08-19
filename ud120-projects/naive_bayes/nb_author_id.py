#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project 

    use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
def author_id(f_train, f_test, l_train, l_test):
	clf = GaussianNB()
	t0 = time()
	clf.fit(f_train, l_train)
	print "training time:", round(time()-t0, 3), "s"	
	t0 = time()
	pred = clf.predict(f_test)
	print "predicting time:", round(time()-t0, 3), "s"
	print 'accuracy  ',  clf.score(f_test, l_test)
	return None


author_id(features_train, features_test, \
	labels_train, labels_test)


#########################################################


