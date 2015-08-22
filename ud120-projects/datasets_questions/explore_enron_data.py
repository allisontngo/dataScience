#!/usr/bin/python

""" 
    starter code for exploring the Enron dataset (emails + finances) 
    loads up the dataset (pickled dict of dicts)

    the dataset has the form
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person
    you should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import numpy as np
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print 'number of people: ', len(enron_data)
print 'number of features: ', max([len(i.keys()) for i in enron_data.itervalues()])
print 'number of poi: ', sum([i[key] for key in i for i in enron_data.itervalues() if key == 'poi'])

names = np.loadtxt('../final_project/poi_names.txt', skiprows=1, dtype=str)
print 'number of poi in txt: ', len(names)

print 'total value of stock belonging to James Prentice: ', \
		enron_data['PRENTICE JAMES']['total_stock_value']

print 'number of messages from Wesley Colwell to poi: ', \
		enron_data['COLWELL WESLEY']['from_this_person_to_poi']

print 'value of stock options for Jeffrey Skilling: ', \
		enron_data['SKILLING JEFFREY K']['exercised_stock_options']

[(i,enron_data[i]['total_payments']) for i in enron_data.iterkeys() \
		if i in ['SKILLING JEFFREY K', 'LAY KENNETH L', 'FASTOW ANDREW S']]

print 'people in dataset with quantified salaries: ', \
		len([i['salary'] for i in enron_data.itervalues() if i['salary'] != 'NaN'])

print 'people in dataset with known email address: ', \
		len([i['email_address'] for i in enron_data.itervalues() if i['email_address'] != 'NaN'])

print 'percentage of people with NaN for total payment: ', \
		len([i['total_payments'] for i in enron_data.itervalues() \
		if i['total_payments'] == 'NaN'])/float(len(enron_data)) * 100

print 'percentage of poi with NaN for total payment: ', \
	len([i['total_payments'] for i in enron_data.itervalues() \
	if (i['total_payments'] == 'NaN') & i['poi']])/float(len(enron_data))

