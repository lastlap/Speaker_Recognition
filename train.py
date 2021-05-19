# import librosa
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import os
import csv
import time

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

# Saving Decision Tree Graph
from six import StringIO  
# from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


# Uncomment these 2 lines to hide warnings
# import warnings
# warnings.filterwarnings('ignore')

# Read Data
data = pd.read_csv('out.csv')
data = data.drop(['Unnamed: 0','file'],axis=1)


# Standardize Data
scaler = StandardScaler()
# choose features you want using data.iloc. Do not use last column with labels.
X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
# X = scaler.fit_transform(np.array(data.iloc[:, 6:16], dtype = float))
y = np.array(data['label'],dtype=int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape,y_train.shape,y_test.shape)

## SVM Classifier

from sklearn import svm

svm_clf = svm.SVC(kernel='linear')
svm_clf.fit(X_train,y_train)
y_pred = svm_clf.predict(X_test)

print("accuracy:",accuracy_score(y_test,y_pred))

## SVM OVR

from sklearn.multiclass import OneVsRestClassifier

clf = OneVsRestClassifier(svm.LinearSVC())
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("accuracy:",accuracy_score(y_test,y_pred))

## SVM OVO

from sklearn.multiclass import OneVsOneClassifier

clf = OneVsOneClassifier(svm.LinearSVC())
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("accuracy:",accuracy_score(y_test,y_pred))

## Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion='gini',max_depth=3)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("accuracy:",accuracy_score(y_test,y_pred))

## Visualize Decision Tree Classifier

feature_cols = list(data.columns)[:-1]
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1','2'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')
# Image(graph.create_png())

## Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=5,max_depth=3,random_state=1)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("accuracy:",accuracy_score(y_test,y_pred))

## Gradient Boosted Tree Classifier

from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=10,max_depth=3)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("accuracy:",accuracy_score(y_test,y_pred))

## AdaBoost Classifier

from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=10)
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("accuracy:",accuracy_score(y_test,y_pred))

## XgBoost Classifier

from xgboost import XGBClassifier

clf = XGBClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print("accuracy:",accuracy_score(y_test,y_pred))

