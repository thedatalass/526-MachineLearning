
# coding: utf-8

# In[2]:

#Indiana University Applied Machine Learning, Spring 2017
#Laura Kahn Final Project
#April 30, 2017

#Code to compare the performance of SVM, Adaboost & Random Forest classifiers on 1996 adult income from Census data-
#want to predict whether a person makes >$50K in income based on one or more features.

#import libraries: dataframe manipulation, machine learning, os tools
import pandas as pd
from pandas import Series, DataFrame
import os
import csv
import matplotlib.pylab as plt
import scipy.stats as stats
import sklearn


# In[3]:

#Change working directory to be the same as where the data set is located
os.chdir("C:/Users/lkahn/Documents/526-AppliedMachineLearning")


# In[4]:

#Load the data
df = pd.read_csv("adult.csv")
#Delete rows with no values or ? in the cell
df_clean = df.dropna()


# In[5]:

#Get summary statistics for the first five rows
df_clean.head(n=5)


# In[7]:

#data types for each feature/variable
df.dtypes


# In[13]:

#Get summary statistics including mean, standard deviation, quartiles
df_clean.describe()


# In[14]:

#Summarize the data
import statsmodels.api as sm
df_clean.std()


# In[6]:

# Encode the categorical features as numbers
#Let's explore the correlation between the different features in the columns.
get_ipython().magic('matplotlib inline')
import numpy as np
import seaborn as sns
from sklearn import preprocessing
def number_encode_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders

# Calculate the correlation and plot it as a heatmap
encoded_data, _ = number_encode_features(df_clean)
sns.heatmap(df.corr(), square=True)
plt.show()


# In[7]:

#We need to preprocess the data to get rid of ? or non-numerical values
import pandas as pd
df_clean_no_missing = df.dropna()
df_clean_no_missing


# In[8]:

import sklearn.cross_validation as cross_validation
import sklearn.metrics as metrics

#Define X, Y variables
X_train, X_test, y_train, y_test = cross_validation.train_test_split(encoded_data[encoded_data.columns - ["Income"]], 
    encoded_data["Income"], train_size=0.80)

#Let's scale the features with mean of 0 and variance of 1 using a Standard Scaler from scikit-learn
scaler = preprocessing.StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train.astype("float64")), columns=X_train.columns)
X_test = scaler.transform(X_test.astype("float64"))


# In[12]:

#Let's look at how Linear Regression classifier does at predicting if income will be greater than $50K
from sklearn.linear_model import LinearRegression
import sklearn.linear_model as linear_model

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
print(regr.coef_)


# In[15]:

#Now let's look at the mean square error of the linear regression model
np.mean((regr.predict(X_test)- y_test)**2)


# In[16]:

#Since a linear regression isn't the right approach since it gives too much weight to data far from the decision frontier,
#we will use Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#Fit a Logistic Regression model to the data
logistic = LogisticRegression()
logistic.fit(X_train, y_train)
print(logistic.coef_)


# In[17]:

#Now let's look at the mean square error of the logistic regression model
np.mean((logistic.predict(X_test)- y_test)**2)


# In[18]:

#Now let's look at accuracy of the Logistic Regression model
from sklearn.metrics import accuracy_score
print (accuracy_score(y_test, logistic.predict(X_test)))


# In[19]:

#Next, let's use a Multinomial Naive Bayes classifier for income prediction
#We need to represent X and Y as count vectors and are going to look at how well the feature 'Education' predicts 'Income'
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(df['Education'])
X.shape


# In[20]:

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()

y = df['Income']

from sklearn.cross_validation import cross_val_score
scores = cross_val_score(mnb, X, y, scoring="accuracy", cv=10)
print("Average accuracy, 10-fold cross validation:")
print(np.mean(scores))


# In[22]:

#Now let's do a Decision Tree Classifier with depth of 3 and leaf size of 5
#We're going to try to hypertune the parameters by increasing the depth size to 5
from sklearn.tree import DecisionTreeClassifier

dt3 = DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=5, min_samples_leaf=5)
dt3.fit(X_train, y_train)

#Now let's look at Decision Tree accuracy
#Now let's look at the accuracy of the Decision Tree Classifier
from sklearn.model_selection import cross_val_score
scores = cross_val_score(dt3, X_test, y_test, scoring="accuracy", cv=10)
print("Average accuracy, 10-fold cross validation:")
print(np.mean(scores))


# In[24]:

#Now we're going to hypertune the cv parameter of the decision tree classifier to 5
from sklearn.tree import DecisionTreeClassifier

dt5 = DecisionTreeClassifier(criterion='gini', random_state=100, max_depth=5, min_samples_leaf=5)
dt5.fit(X_train, y_train)

#Now let's look at Decision Tree accuracy
#Now let's look at the accuracy of the Decision Tree Classifier
from sklearn.model_selection import cross_val_score
scores = cross_val_score(dt5, X_test, y_test, scoring="accuracy", cv=5)
print("Average accuracy, 5-fold cross validation:")
print(np.mean(scores))

#We can see that increasing the number of leaves has no effect, increasing the depth size has no effect on accuracy
#Decreasing the CV from 10 to 5 decreased the accuracy of the Decision Tree model


# In[25]:

#Next, we'll do a k-Nearest Neighbor model for income prediction 
from sklearn.neighbors import KNeighborsClassifier
kNN3 = KNeighborsClassifier(n_neighbors=3)

kNN3.fit(X_train, y_train)

#Now let's look at the accuracy of a KNN with k = 3
from sklearn.model_selection import cross_val_score
scores = cross_val_score(kNN3, X_test, y_test, scoring="accuracy", cv=10)
print("Average accuracy, 10-fold cross validation:")
print(np.mean(scores))


# In[26]:

#Let's try to hypertune the k parameter of the K-Nearest Neighbor model to k=10 to see if we can get 
#better accuracy results 

from sklearn.neighbors import KNeighborsClassifier
kNN10 = KNeighborsClassifier(n_neighbors=10)

kNN10.fit(X_train, y_train)

#Now let's look at the accuracy of a KNN with k = 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(kNN10, X_test, y_test, scoring="accuracy", cv=10)
print("Average accuracy, 10-fold cross validation:")
print(np.mean(scores))


# In[27]:

#Let's try to hypertune the k parameter to k=20 to see if we can get better accuracy results from our kNN model

from sklearn.neighbors import KNeighborsClassifier
kNN20 = KNeighborsClassifier(n_neighbors=20)

kNN20.fit(X_train, y_train)

#Now let's look at the accuracy of a KNN with k = 20
from sklearn.model_selection import cross_val_score
scores = cross_val_score(kNN20, X_test, y_test, scoring="accuracy", cv=10)
print("Average accuracy, 10-fold cross validation:")
print(np.mean(scores))


# In[9]:

#Let's look at how Support Vector Machine classifier does at predicting if income >$50K
from sklearn import svm

svmcl#Let's try to hypertune the k parameter to k=20 to see if we can get better accuracy results from our kNN model

from sklearn.neighbors import KNeighborsClassifier
kNN20 = KNeighborsClassifier(n_neighbors=20)

kNN20.fit(X_train, y_train)

#Now let's look at the accuracy of a KNN with k = 20
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_test, y_test, scoring="accuracy", cv=10)
print("Average accuracy, 10-fold cross validation:")
print(np.mean(scores))f = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
svmclf.fit(X_train,y_train)

#Now let's look at accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(svmclf, X_test, y_test, scoring="accuracy", cv=10)
print("Average accuracy, 10-fold cross validation:")
print(np.mean(scores))


# In[21]:

#Now, let's try tuning the value of the "C" hyperparameter with C=5
#Remember that changing c may or may not produce a different hyperplane
from sklearn import svm

svmclf = svm.SVC(C=5.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
svmclf.fit(X_train,y_train)

#Now let's look at accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(svmclf, X_test, y_test, scoring="accuracy", cv=10)
print("Average accuracy, 10-fold cross validation:")
print(np.mean(scores))


# In[34]:

#Now, let's try tuning the value of the "C" hyperparameter with C=10
#Remember that changing c may or may not produce a different hyperplane
from sklearn import svm

svmclf = svm.SVC(C=10.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
svmclf.fit(X_train,y_train)

#Now let's look at accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(svmclf, X_test, y_test, scoring="accuracy", cv=10)
print("Average accuracy, 10-fold cross validation:")
print(np.mean(scores))


# In[23]:

#Next, let's do an Adaboost classifier with tree depth of 1
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200)
bdt.fit(X_train,y_train)

#Now let's look at accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(bdt, X_test, y_test, scoring="accuracy", cv=10)
print("Average accuracy, 10-fold cross validation:")
print(np.mean(scores))


# In[24]:

#Now let's see what would happen if we change the tree depth of the Adaboost classifier to 5
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5), algorithm="SAMME", n_estimators=200)
bdt.fit(X_train,y_train)

#Now let's look at accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(bdt, X_test, y_test, scoring="accuracy", cv=10)
print("Average accuracy, 10-fold cross validation:")
print(np.mean(scores))


# In[30]:

#Now let's look at a Random Forest Classifier with tree depth of 1
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=1)
rf.fit(X_train,y_train)

#Now let's look at accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf, X_test, y_test, scoring="accuracy", cv=10)
print("Average accuracy, 10-fold cross validation:")
print(np.mean(scores))


# In[31]:

#Now let's look at a Random Forest Classifier with tree depth of 10
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=10)
rf.fit(X_train,y_train)

#Now let's look at accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf, X_test, y_test, scoring="accuracy", cv=10)
print("Average accuracy, 10-fold cross validation:")
print(np.mean(scores))


# In[32]:

#Now let's look at a Random Forest Classifier with tree depth of 20
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=20)
rf.fit(X_train,y_train)

#Now let's look at accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf, X_test, y_test, scoring="accuracy", cv=10)
print("Average accuracy, 10-fold cross validation:")
print(np.mean(scores))


# In[33]:

#Now let's look at a Random Forest Classifier with tree depth of 15
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=15)
rf.fit(X_train,y_train)

#Now let's look at accuracy
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rf, X_test, y_test, scoring="accuracy", cv=10)
print("Average accuracy, 10-fold cross validation:")
print(np.mean(scores))


# In[39]:

#Accuracy was used because it is the number of correct predictions made as a ratio of all predictions made.
#It is the most common evaluation metric for classification problems.

#Adaboost classifier with tree depth of 5 had the highest accuracy at .8561, then Random Forest with tree depth of 10 
# = 0.8535 accuracy, then SVM with C=5 had a 0.8447 accuracy, then Decision Tree with depth of 3 = 0.8446, then 
#Logistic Regression = 0.8239, then kNN10 = 0.8207, then Multinomial Naive Bayes = 0.7796.

#Each of the hyperparemeters C and tree depth were optimized for improved accuracy

#Next, we're going to plot the accuracies for each model for a visual comparison of the results.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

labels = ['SVM5', 'Ada5', 'RF10', 'DT3', 'kNN10', 'LR', 'MNB']
data = [84.47, 85.61, 85.35, 84.46, 82.07, 82.39, 77.96]

xlocations = np.array(range(len(data)))+0.5
width = 0.5
bar(xlocations, data, width=width, color='green')
yticks(fontsize=16)
xticks(xlocations+ width/2, labels, fontsize=16, rotation='vertical', color='purple')
xlim(0, xlocations[-1]+width*2)
ylim(77,87)

#Add title to plot
title('Machine Learning Classifier Accuracy for Income Prediction', fontsize=16, color='purple')

gca().get_xaxis().tick_bottom()
gca().get_yaxis().tick_left()
show()

