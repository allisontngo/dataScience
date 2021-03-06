#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        clean away the 10% of points that have the largest
        residual errors (different between the prediction
        and the actual net worth)

        return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error)
    """
    
    cleaned_data = []

    ### your code goes here
    residual = abs(predictions - net_worths)
    cleaned_data = zip(ages.flatten(), net_worths.flatten(), residual.flatten())
    cleaned_data.sort(key = lambda x: x[2])
    return cleaned_data[:int(len(residual)*-0.1)]

