from strkernel.gappy_kernel import gappypair_kernel as gk
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import pandas as pd
import numpy as np

X = pd.read_csv('data/Xtr0.csv')['seq']\
.append(pd.read_csv('data/Xtr1.csv')['seq'])\
.append(pd.read_csv('data/Xtr2.csv')['seq'])

y = pd.read_csv('data/Ytr0.csv')['Bound']\
.append(pd.read_csv('data/Ytr1.csv')['Bound'])\
.append(pd.read_csv('data/Ytr2.csv')['Bound']).values

X = gk(X, k=7, t=0, g=2)

kf = KFold(n_splits=5, random_state=5)
kf.get_n_splits(X)

for i in [0.1,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.5,2.0]:
    res = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        gappyclassifier = SVC(C=i, kernel='linear')
        gappyclassifier.fit(X_train, y_train)
        y_pred = gappyclassifier.predict(X_test)
        
        res.append(accuracy_score(y_test, y_pred))
    print(np.mean(res))

from sklearn.preprocessing import normalize

def normK(X, Y):
    X_n = normalize(X, norm='l2', axis=1)
    Y_n = normalize(Y, norm='l2', axis=1)
    K = np.dot(X_n, Y_n.T)
    return K

kf = KFold(n_splits=5, random_state=5)
kf.get_n_splits(X)

for i in [0.1,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.5,2.0]:
    res = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        normlassifier = SVC(C=i, kernel=normK) 
        normlassifier.fit(X_train, y_train)  
        y_pred = normlassifier.predict(X_test).astype(int)

        res.append(accuracy_score(y_test, y_pred))
    print(np.mean(res))
