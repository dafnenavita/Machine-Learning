
# coding: utf-8

# In[ ]:


import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import metrics

#load digits data set 
from sklearn.datasets import load_digits
digits = load_digits()

#train test split - train data is sent to grid search , test data is used for predictions and confusion matrix
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

#parameters
parameters_svm = {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['linear','rbf','poly'], 'degree': [1 ,2 ,3] }
parameters_dt = {'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] , 'min_samples_split': [2, 5, 6, 7, 10] ,'max_features' : ['auto', 'sqrt'], 'min_samples_leaf': [1, 2, 4, 6, 8]}
parameters_nn = {'activation' : ['identity', 'logistic', 'tanh', 'relu'],'solver' : ['lbfgs', 'sgd', 'adam'],
                'learning_rate' : ['constant', 'invscaling', 'adaptive'], 'early_stopping' : [True], 'max_iter' :[100, 1000], 'alpha' :[0.0001, 0.01]}
parameters_gnb = {'priors' : [None]}
parameters_lr = {'penalty' : ['l1','l2'],'C': [1, 10, 100, 1000],'fit_intercept' : [True,False],'class_weight' : ['balanced', None]} 
parameters_knn = {'n_neighbors' : [5 , 3 , 1],'weights' : ['uniform'],'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],'p' : [1 ,2]}

#parameter array   
params = [parameters_svm , parameters_dt , parameters_nn, parameters_gnb, parameters_lr, parameters_knn]

#classifiers 
svm = SVC()
dt = DecisionTreeClassifier()
mlp = MLPClassifier()
gnb = GaussianNB()
logReg = LogisticRegression()
knn = KNeighborsClassifier()



#classifiers array
classifiers = [svm,dt,mlp,gnb,logReg,knn]

#iterate over each classifier and corresponding parameters 
for i in range(len(params)):
    gs = GridSearchCV(classifiers[i], params[i])
    gs = gs.fit(x_train, y_train)
    print("================Classifiers============")
    print(classifiers[i])
    sorted(gs.cv_results_.keys())
    print ("============Best Parameters=============")
    print(gs.best_params_)
    print ("============Best Score=============")
    print(gs.best_score_)
    scores = cross_val_score(gs, x_train, y_train, cv=5)
    print('==============Scores====================')
    print(scores) 
    print('==============Accuracy==================')
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    predictions = gs.predict(x_test)
    cm = metrics.confusion_matrix(y_test, predictions)
    print('=============Confusion matrix==============')
    print(cm)
    print('==========classification report=================')
    print(metrics.classification_report(y_test, predictions))
    print('\n***********************************************************')
    print("\n")
    
print('==============Best Estimator===========')   
print(gs.best_estimator_)








