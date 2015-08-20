#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 2 (SVM) mini-project

    use an SVM to identify emails from the Enron corpus by their authors
    
    Sara has label 0
    Chris has label 1

"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn import svm
from sklearn.metrics import accuracy_score
features_train1 = features_train[:len(features_train)/100]
labels_train1 = labels_train[:len(labels_train)/100]
def clf_pred(f_train, l_train, **kwargs):
	c = kwargs.get('C', None)
	k = kwargs.get('kernel','rbf')	
	clf = svm.SVC(kernel=k) if c is None else svm.SVC(kernel=k, C=c)
	t0 = time()
	clf.fit(f_train, l_train)
	print "training time:", round(time()-t0, 3), "s"
	t0 = time()
	pred = clf.predict(features_test)
	print "prediction time:", round(time()-t0, 3), "s"
	return pred

def accuracy(pred):
	#linear accuracy about .88
	#accuracy_score is faster than clf.score(features_test, labels_test)
	return 'accuracy: %f' % accuracy_score(pred, labels_test)

#find optimal C for rbf kernel
#print 'C=10 \n', accuracy(clf_pred(C=10))
#print 'C=100 \n', accuracy(clf_pred(C=100))
#print 'C=1000 \n', accuracy(clf_pred(C=1000))
pred = clf_pred(features_train1, labels_train1, C=10000)
print 'C=10000 \n', accuracy(pred) #highest accuracy

#find element 10, 26, 50 using subset of training set
print (pred[10], pred[26], pred[50])

pred_full = clf_pred(features_train, labels_train, C=10000)
print 'predicted to be 1: ', sum(pred_full)
#########################################################


