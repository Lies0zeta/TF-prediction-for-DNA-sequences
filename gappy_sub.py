from strkernel.gappy_kernel import gappypair_kernel as gk
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import pandas as pd
import numpy as np

x_df_train = pd.read_csv('data/Xtr0.csv')['seq']\
    .append(pd.read_csv('data/Xtr1.csv')['seq'])\
    .append(pd.read_csv('data/Xtr2.csv')['seq'])
x_df_test = pd.read_csv('data/Xte0.csv')['seq']\
    .append(pd.read_csv('data/Xte1.csv')['seq'])\
    .append(pd.read_csv('data/Xte2.csv')['seq'])
X = x_df_train.append(x_df_test)

y_df_train = pd.read_csv('data/Ytr0.csv')['Bound']\
    .append(pd.read_csv('data/Ytr1.csv')['Bound'])\
    .append(pd.read_csv('data/Ytr2.csv')['Bound'])
# y_df_more_train = pd.read_csv('data/better.csv')['Bound']
y_df = y_df_train#.append(y_df_more_train)
y = y_df.values

X = gk(X, k=7, t=0, g=2)

indices_train = range(0,6000)
X_train = X[indices_train]
X_test = X[6000:9000]
y_train = y[indices_train]

y_pred = []
for i in [0.1,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.5,2.0]:
    gappyclassifier = SVC(C=i, kernel='linear')
    gappyclassifier.fit(X_train, y_train)
    y_pred.append(gappyclassifier.predict(X_test))

#     print(accuracy_score(y_test, y_pred[i]))

dataset = pd.DataFrame({'Id' : range(3000), 'Bound' : y_pred[0]})\
    .set_index('Id')
dataset.to_csv('res/pred_gappy_linear.csv', index=True)

from sklearn.preprocessing import normalize

def normK(X, Y):
    X_n = normalize(X, norm='l2', axis=1)
    Y_n = normalize(Y, norm='l2', axis=1)
    K = np.dot(X_n, Y_n.T)
    return K

y_pred = []
for i in [0.1,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.5,2.0]:
    gappyclassifier = SVC(C=i, kernel=normK)
    gappyclassifier.fit(X_train, y_train)
    y_pred.append(gappyclassifier.predict(X_test))

#     print(accuracy_score(y_test, y_pred[i]))

dataset = pd.DataFrame({'Id' : range(3000), 'Bound' : y_pred[5]})\
    .set_index('Id')
dataset.to_csv('res/pred_gappy_spectrum.csv', index=True)
