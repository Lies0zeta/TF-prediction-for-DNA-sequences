#####   Lies Hadjadj & Narek Davtyan    ####
##### Advanced Learning Model project   ####

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC 
from sklearn.preprocessing import normalize
import pandas as pd
import numpy as np


# Reading the dataset
X = pd.read_csv('Xtr0.csv')['seq']\
.append(pd.read_csv('Xtr1.csv')['seq'])\
.append(pd.read_csv('Xtr2.csv')['seq'])

Y = pd.read_csv('Ytr0.csv')['Bound']\
.append(pd.read_csv('Ytr1.csv')['Bound'])\
.append(pd.read_csv('Ytr2.csv')['Bound']).values

X_test = pd.read_csv('Xte0.csv')['seq']\
.append(pd.read_csv('Xte1.csv')['seq'])\
.append(pd.read_csv('Xte2.csv')['seq'])


# function to convert sequence strings into k-mer words, default size = 7
def getKmers(sequence, size=7):
    return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]

# Computing the k-mer representation
X = list(X.apply(lambda x: getKmers(x)))
X_test = list(X_test.apply(lambda x: getKmers(x)))

for item in range(len(X)):
    X[item] = ' '.join(X[item])

for item in range(len(X_test)):
    X_test[item] = ' '.join(X_test[item])

X_train = X.copy()
X_train.extend(X_test)
cv = CountVectorizer(ngram_range=(4,4))
cv.fit(X_train)
X = cv.transform(X)
X_test = cv.transform(X_test)

# the normalized spectrum kernel function
def normK(X, Y):
    X_n = normalize(X, norm='l2', axis=1)
    Y_n = normalize(Y, norm='l2', axis=1)
    K = np.dot(X_n, Y_n.T)
    return K

#### SVM implementation, we prefered to not use it in this submission for its computational complexity
#clf = KSVM(kernel="normK")
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)

# we used the predefined SVM from scikit-learn library for performance, C=0.9 was 5-folds cross validated 
svclassifier = SVC(C=0.9, kernel=normK) 
svclassifier.fit(X, Y)  

# predicting the test set
y_pred = svclassifier.predict(X_test)

# storing the prediction
dataset = pd.DataFrame({'Id':range(3000),'Bound':y_pred})
dataset.to_csv('Yte.csv', index=False)