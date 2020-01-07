# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 01:18:37 2019

@author: M_M
"""

from sklearn.neural_network import MLPClassifier
import random
#Q1
def setData():
    data_set = []
    Y_0 = [0, 1, 1, 1, 0, 
           0, 1, 0, 1, 0, 
           0, 1, 0, 1, 0, 
           0, 1, 0, 1, 0, 
           0, 1, 0, 1, 0, 
           0, 1, 1, 1, 0]
    data_set.append(Y_0)
    Y_1 = [0, 0, 1, 0, 0, 
           0, 0, 1, 0, 0, 
           0, 0, 1, 0, 0, 
           0, 0, 1, 0, 0, 
           0, 0, 1, 0, 0, 
           0, 0, 1, 0, 0]
    data_set.append(Y_1)
    Y_2 = [0, 1, 1, 1, 0, 
           0, 0, 0, 1, 0, 
           0, 1, 1, 1, 0, 
           0, 1, 0, 0, 0, 
           0, 1, 0, 0, 0, 
           0, 1, 1, 1, 0]
    data_set.append(Y_2)
    Y_3 = [0, 1, 1, 1, 0, 
           0, 0, 0, 1, 0, 
           0, 1, 1, 1, 0, 
           0, 0, 0, 1, 0, 
           0, 0, 0, 1, 0, 
           0, 1, 1, 1, 0]
    data_set.append(Y_3)
    Y_4 = [0, 1, 0, 1, 0, 
           0, 1, 0, 1, 0, 
           0, 1, 1, 1, 0, 
           0, 0, 0, 1, 0, 
           0, 0, 0, 1, 0, 
           0, 0, 0, 1, 0]
    data_set.append(Y_4)
    Y_5 = [0, 1, 1, 1, 0, 
           0, 1, 0, 0, 0, 
           0, 1, 0, 0, 0, 
           0, 1, 1, 1, 0, 
           0, 0, 0, 1, 0, 
           0, 1, 1, 1, 0]
    data_set.append(Y_5)
    Y_6 = [0, 1, 1, 1, 0, 
           0, 1, 0, 0, 0, 
           0, 1, 0, 0, 0, 
           0, 1, 1, 1, 0, 
           0, 1, 0, 1, 0, 
           0, 1, 1, 1, 0]
    data_set.append(Y_6)
    Y_7 = [0, 1, 1, 1, 0, 
           0, 0, 0, 1, 0, 
           0, 0, 0, 1, 0, 
           0, 0, 0, 1, 0, 
           0, 0, 0, 1, 0, 
           0, 0, 0, 1, 0]
    data_set.append(Y_7)
    Y_8 = [0, 1, 1, 1, 0, 
           0, 1, 0, 1, 0, 
           0, 1, 1, 1, 0, 
           0, 1, 1, 1, 0, 
           0, 1, 0, 1, 0, 
           0, 1, 1, 1, 0]
    data_set.append(Y_8)
    Y_9 = [0, 1, 1, 1, 0, 
           0, 1, 0, 1, 0, 
           0, 1, 1, 1, 0, 
           0, 0, 0, 1, 0, 
           0, 0, 0, 1, 0, 
           0, 1, 1, 1, 0]
    data_set.append(Y_9)
    return data_set
#Q2
N_set = [0, 1, 5, 10, 15]
dataset_N = []
dataset_N.append(setData())
for i in range(1,len(N_set)):
    dataset_copy = setData()
    for j in range(N_set[i]):
        selected_pixel = random.randint(0, 29)
        for num_set in dataset_copy:
            if num_set[selected_pixel] == 0:
                num_set[selected_pixel] = 1
            else:
                num_set[selected_pixel] = 0     
    dataset_N.append(dataset_copy)

#Q3
error = 0   
generate_set = [(0,2), (1,2), (2,2), (3,2), (4,2)] 

note = []
note.append("With dataset_0 for the training, dataset_5 for the test, and one hidden layer:")
note.append("With dataset_1 for the training, dataset_5 for the test, and one hidden layer:")
note.append("With dataset_5 for the training, dataset_5 for the test, and one hidden layer:")
note.append("With dataset_10 for the training, dataset_5 for the test, and one hidden layer:")
note.append("With dataset_15 for the training, dataset_5 for the test, and one hidden layer:")
note.append("With dataset_0 for the training, dataset_5 for the test, and two hidden layer:")
note.append("With dataset_1 for the training, dataset_5 for the test, and two hidden layer:")
note.append("With dataset_5 for the training, dataset_5 for the test, and two hidden layer:")
note.append("With dataset_10 for the training, dataset_5 for the test, and two hidden layer:")
note.append("With dataset_15 for the training, dataset_5 for the test, and two hidden layer:")

for k in range(len(generate_set)*2):   
    if k < 5:
        train = generate_set[k][0]
        test = generate_set[k][1]
    else:
        train = generate_set[k-len(generate_set)][0]
        test = generate_set[k-len(generate_set)][1]
    print(note[k])
    X_train = dataset_N[train]
    X_test = dataset_N[test]
    y = [0,1,2,3,4,5,6,7,8,9]
    if k < 5:
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(30,))
    else:
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(30,30))
    clf.fit(X_train, y)                         
    clf.predict(X_test)
    p = clf.predict_proba(X_test)
    error_count = 0
    for i in range(len(y)):
        name = str(y[i])
        maxProbability = 0
        predict_result = ''
        for j in range(len(y)):
            num = str(j)
            if p[i][j] > maxProbability:
                maxProbability = p[i][j]
                predict_result = num
        print("The max probability of prediction of input " + name + " equals to " + predict_result + " is: ", maxProbability)
        if predict_result != name:
            error_count += 1
    error += error_count
    print("total error of this prediction:", error_count)
    print()
#compute average error
error_avg = error/(len(y)*len(N_set))
print("The average of error is:", error_avg)
print()


#    
##With dataset_1 for the training, dataset_5 for the test, and one hidden layer
#error_count = 0
#train = 1
#test = 2
#X_train = dataset_N[train]
#X_test = dataset_N[test]
#y = [0,1,2,3,4,5,6,7,8,9]
#clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(30,))
#clf.fit(X_train, y)                         
#clf.predict(X_test)
#p = clf.predict_proba(X_test)
#for i in range(len(y)):
#    name = str(y[i])
#    maxProbability = 0
#    predict_result = ''
#    for j in range(len(y)):
#        num = str(j)
#        if p[i][j] > maxProbability:
#            maxProbability = p[i][j]
#            predict_result = num
#        print("The probability of " + name + " equals to " + num + " is: ", p[i][j])
#    if predict_result != name:
#        error_count += 1
#    print()
#error += error_count
#print("total error of this prediction:", error_count)
#print("\n\n")
##print(error)
#
##With dataset_5 for the training, dataset_5 for the test, and one hidden layer
#error_count = 0
#train = 2
#test = 2
#X_train = dataset_N[train]
#X_test = dataset_N[test]
#y = [0,1,2,3,4,5,6,7,8,9]
#clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(30,))
#clf.fit(X_train, y)                         
#clf.predict(X_test)
#p = clf.predict_proba(X_test)
#for i in range(len(y)):
#    name = str(y[i])
#    maxProbability = 0
#    predict_result = ''
#    for j in range(len(y)):
#        num = str(j)
#        if p[i][j] > maxProbability:
#            maxProbability = p[i][j]
#            predict_result = num
#        print("The probability of " + name + " equals to " + num + " is: ", p[i][j])
#    if predict_result != name:
#        error_count += 1
#    print()
#error += error_count
#print("total error of this prediction:", error_count)
#print("\n\n")
##print(error)
#
##With dataset_10 for the training, dataset_5 for the test, and one hidden layer
#error_count = 0 
#train = 3
#test = 2
#X_train = dataset_N[train]
#X_test = dataset_N[test]
#y = [0,1,2,3,4,5,6,7,8,9]
#clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(30,))
#clf.fit(X_train, y)                         
#clf.predict(X_test)
#p = clf.predict_proba(X_test)
#for i in range(len(y)):
#    name = str(y[i])
#    maxProbability = 0
#    predict_result = ''
#    for j in range(len(y)):
#        num = str(j)
#        if p[i][j] > maxProbability:
#            maxProbability = p[i][j]
#            predict_result = num
#        print("The probability of " + name + " equals to " + num + " is: ", p[i][j])
#    if predict_result != name:
#        error_count += 1
#    print()
#error += error_count
#print("total error of this prediction:", error_count)
#print("\n\n")
##print(error)
#    
##With dataset_15 for the training, dataset_5 for the test, and one hidden layer
#error_count = 0 
#train = 4
#test = 2
#X_train = dataset_N[train]
#X_test = dataset_N[test]
#y = [0,1,2,3,4,5,6,7,8,9]
#clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(30,))
#clf.fit(X_train, y)                         
#clf.predict(X_test)
#p = clf.predict_proba(X_test)
#for i in range(len(y)):
#    name = str(y[i])
#    maxProbability = 0
#    predict_result = ''
#    for j in range(len(y)):
#        num = str(j)
#        if p[i][j] > maxProbability:
#            maxProbability = p[i][j]
#            predict_result = num
#        print("The probability of " + name + " equals to " + num + " is: ", p[i][j])
#    if predict_result != name:
#        error_count += 1
#    print()
#error += error_count
#print("total error of this prediction:", error_count)
#print("\n\n")
##print(error)
#
##With dataset_0 for the training, dataset_5 for the test, and two hidden layer
#error_count = 0
#train = 0
#test = 2
#X_train = dataset_N[train]
#X_test = dataset_N[test]
#y = [0,1,2,3,4,5,6,7,8,9]
#clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(30,30,))
#clf.fit(X_train, y)                         
#clf.predict(X_test)
#p = clf.predict_proba(X_test)
#for i in range(len(y)):
#    name = str(y[i])
#    maxProbability = 0
#    predict_result = ''
#    for j in range(len(y)):
#        num = str(j)
#        if p[i][j] > maxProbability:
#            maxProbability = p[i][j]
#            predict_result = num
#        print("The probability of " + name + " equals to " + num + " is: ", p[i][j])
#    if predict_result != name:
#        error_count += 1
#    print()
#error += error_count
#print("total error of this prediction:", error_count)
#print("\n\n")
##print(error)
#    
##With dataset_1 for the training, dataset_5 for the test, and two hidden layer
#error_count = 0
#train = 1
#test = 2
#X_train = dataset_N[train]
#X_test = dataset_N[test]
#y = [0,1,2,3,4,5,6,7,8,9]
#clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(30,30,))
#clf.fit(X_train, y)                         
#clf.predict(X_test)
#p = clf.predict_proba(X_test)
#for i in range(len(y)):
#    name = str(y[i])
#    maxProbability = 0
#    predict_result = ''
#    for j in range(len(y)):
#        num = str(j)
#        if p[i][j] > maxProbability:
#            maxProbability = p[i][j]
#            predict_result = num
#        print("The probability of " + name + " equals to " + num + " is: ", p[i][j])
#    if predict_result != name:
#        error_count += 1
#    print()
#error += error_count
#print("total error of this prediction:", error_count)
#print("\n\n")
##print(error)
#
##With dataset_5 for the training, dataset_5 for the test, and two hidden layer
#error_count = 0
#train = 2
#test = 2
#X_train = dataset_N[train]
#X_test = dataset_N[test]
#y = [0,1,2,3,4,5,6,7,8,9]
#clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(30,30))
#clf.fit(X_train, y)                         
#clf.predict(X_test)
#p = clf.predict_proba(X_test)
#for i in range(len(y)):
#    name = str(y[i])
#    maxProbability = 0
#    predict_result = ''
#    for j in range(len(y)):
#        num = str(j)
#        if p[i][j] > maxProbability:
#            maxProbability = p[i][j]
#            predict_result = num
#        print("The probability of " + name + " equals to " + num + " is: ", p[i][j])
#    if predict_result != name:
#        error_count += 1
#    print()
#error += error_count
#print("total error of this prediction:", error_count)
#print("\n\n")
##print(error)
#
##With dataset_10 for the training, dataset_5 for the test, and two hidden layer 
#error_count = 0
#train = 3
#test = 2
#X_train = dataset_N[train]
#X_test = dataset_N[test]
#y = [0,1,2,3,4,5,6,7,8,9]
#clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(30,30))
#clf.fit(X_train, y)                         
#clf.predict(X_test)
#p = clf.predict_proba(X_test)
#for i in range(len(y)):
#    name = str(y[i])
#    maxProbability = 0
#    predict_result = ''
#    for j in range(len(y)):
#        num = str(j)
#        if p[i][j] > maxProbability:
#            maxProbability = p[i][j]
#            predict_result = num
#        print("The probability of " + name + " equals to " + num + " is: ", p[i][j])
#    if predict_result != name:
#        error_count += 1
#    print()
#error += error_count
#print("total error of this prediction:", error_count)
#print("\n\n")
##print(error)
##    
###With dataset_15 for the training, dataset_5 for the test, and one hidden layer
#error_count = 0
#train = 4
#test = 2
#X_train = dataset_N[train]
#X_test = dataset_N[test]
#y = [0,1,2,3,4,5,6,7,8,9]
#clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(30,30))
#clf.fit(X_train, y)                         
#clf.predict(X_test)
#p = clf.predict_proba(X_test)
#for i in range(len(y)):
#    name = str(y[i])
#    maxProbability = 0
#    predict_result = ''
#    for j in range(len(y)):
#        num = str(j)
#        if p[i][j] > maxProbability:
#            maxProbability = p[i][j]
#            predict_result = num
#        print("The probability of " + name + " equals to " + num + " is: ", p[i][j])
#    if predict_result != name:
#        error_count += 1
#    print()
#error += error_count
#print("total error of this prediction:", error_count)
#print("\n\n")
##print(error)
#
##compute average error
#error_avg = error/(len(y)*len(N_set))
#print("The average of error is:", error_avg)