from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from pyfm import pylibfm
import pandas as pd
import numpy as np

def getKmers(sequence, size=7):
    return [sequence[x:x+size].lower() \
        for x in range(len(sequence) - size + 1)]


x_df = pd.read_csv('data/Xtr0.csv')\
    .append(pd.read_csv('data/Xtr1.csv'))\
    .append(pd.read_csv('data/Xtr2.csv'))

y_df = pd.read_csv('data/Ytr0.csv')\
    .append(pd.read_csv('data/Ytr1.csv'))\
    .append(pd.read_csv('data/Ytr2.csv'))

xy_df = x_df.set_index('Id').join(y_df.set_index('Id'))
xy_df['words'] = xy_df.apply(lambda x: getKmers(x['seq']), axis=1)
xy_df = xy_df.drop('seq', axis=1)

from sklearn.feature_extraction.text import CountVectorizer
readable_words = list(xy_df['words'])
for item in range(len(readable_words)):
    readable_words[item] = ' '.join(readable_words[item])
cv = CountVectorizer(ngram_range=(4,4))
X = cv.fit_transform(readable_words)
y = xy_df.iloc[:, 0].values

kf = KFold(n_splits=5, random_state=111)
kf.get_n_splits(X)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    fm = pylibfm.FM(num_factors=2000, num_iter=5, \
                verbose=True, task="classification", \
                initial_learning_rate=0.0001, \
                learning_rate_schedule="optimal")
    fm.fit(X_train.astype(float), y_train)
    
    y_pred__ = fm.predict(X_test)
    y_pred_ = fm._prepare_y(y_pred__)
    y_pred = [0 if x==-1. else 1 for x in y_pred_]
    
    print(accuracy_score(y_test, y_pred))
