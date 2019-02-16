#####   Lies Hadjadj & Narek Davtyan    ####
##### Advanced Learning Model project   ####

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC 
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np
from KSVM import KSVM

# Reading the dataset
X = pd.read_csv('data/Xtr0.csv')['seq']#\
#.append(pd.read_csv('data/Xtr1.csv')['seq'])\
#.append(pd.read_csv('data/Xtr2.csv')['seq'])

Y = pd.read_csv('data/Ytr0.csv')['Bound'].values#\
#.append(pd.read_csv('data/Ytr1.csv')['Bound'])\
#.append(pd.read_csv('data/Ytr2.csv')['Bound']).values


# function to convert sequence strings into k-mer words, default size = 7
def getKmers(sequence, size=7):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

# Computing the k-mer representation
X = list(X.apply(lambda x: getKmers(x)))

for item in range(len(X)):
    X[item] = ' '.join(X[item])

cv = CountVectorizer(ngram_range=(4,4))
X = cv.fit_transform(X)

# split the data for train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.40, random_state=42)

#### SVM implementation, we prefered to not use it in this submission for its computational complexity
clf = KSVM(kernel="linear")
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# we used the predefined SVM from scikit-learn library for performance, C=0.9 was 5-folds cross validated 

# the normalized spectrum kernel function
# def normK(X, Y):
#     X_n = normalize(X, norm='l2', axis=1)
#     Y_n = normalize(Y, norm='l2', axis=1)
#     K = np.dot(X_n, Y_n.T)
#     return K

# svclassifier = SVC(C=0.9, kernel=normK) 
# svclassifier.fit(X_train, y_train)  

# predicting the test set
#y_pred = svclassifier.predict(X_test)

# score of the prediction
print(accuracy_score(y_test, y_pred))
