import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pplt
import sys

#################################################################
### Definitions

def import_data(instr):
    training_data = []
    test_data = []
    reading_test_data = False
    with open(instr) as file:
        for i, line in enumerate(file):
            words = line.split()
            if i == 0:
                continue
            elif len(words) == 1:
                print 'done training data'
                reading_test_data = True
                curr_data = []
            elif reading_test_data:
                params = [float(val.split(':')[1]) for val in words[1:]]
                test_data.append(params)
            else:
                params = [float(val.split(':')[1]) for val in words[2:]]
                params += [float(words[1])]
                training_data.append(params)
    train = np.array(training_data)
    X = train[:,:-1]
    y = train[:,-1]
    test = np.array(test_data)
    return X,y,test

def normalize_cols(arr):
    arr_ = np.copy(arr)
    means = np.mean(arr_, axis=1)
    for i,row in enumerate(arr_.T):
        row -= means[i]
    stds = np.std(arr_, axis=1)
    for i,row in enumerate(arr_.T):
        row /= stds[i]
    return arr_

def gini(bucket):
    total = len(bucket)
    pluses = np.sum(bucket > 0)/total
    minuses = np.sum(bucket < 0)/total
    vec = np.array([pluses, minuses])
    return 1 - np.dot(vec, vec)

#################################################################
### Script

if __name__ == "__main__":
    X, y, X_test = import_training_data("inputB")
    X_ = normalize_cols(X)
