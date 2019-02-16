#####   Lies Hadjadj & Narek Davtyan    ####
##### Advanced Learning Model project   ####

from sklearn.preprocessing import normalize
from scipy.sparse import lil_matrix
from scipy.sparse import csr_matrix
import numpy as np 
import cvxopt
import cvxopt.solvers

cvxopt.solvers.options['show_progress'] = False

class KSVM():
    def __init__(self,kernel="rbf",polyconst=1,gamma=10,degree=2):
        self.kernel = kernel
        self.polyconst = float(polyconst)
        self.gamma = float(gamma)
        self.degree = degree
        self.kf = {
            "linear":self.linear,
            "rbf":self.rbf,
            "poly":self.polynomial,
            "normK":self.normK
        }
        self._support_vectors = None
        self._alphas = None
        self.intercept = None
        self._n_support = None
        self.weights = None
        self._support_labels = None
        self._indices = None

    def linear(self,x,y):
        return x.T.dot(y)[0,0]

    def polynomial(self,x,y):
        return (x.T.dot(y).data + self.polyconst)**self.degree

    def rbf(self,x,y):
        return np.exp(-1.0*self.gamma*np.dot(np.subtract(x,y).T,np.subtract(x,y)))
    
    def normK(self,x,y):
        x_n = normalize(x, norm='l2', axis=1)
        y_n = normalize(y, norm='l2', axis=1)
        return x_n.T.dot(y_n)[0,0]

    def transform(self,X):
        K = lil_matrix((X.shape[0],X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                K[i,j] = self.kf[self.kernel](X[i],X[j])
        return K
    
    def scipy_sparse_to_spmatrix(self,A):
        coo = A.tocoo()
        SP = cvxopt.spmatrix(coo.data.tolist(), coo.row.tolist(), coo.col.tolist(), size=A.shape)
        return SP
    
    def fit(self,data,labels):
        num_data, num_features = data.shape
        labels = labels.astype(np.double)
        K = self.transform(data)
        P = self.scipy_sparse_to_spmatrix(K.multiply(lil_matrix(np.outer(labels,labels))))
        q = cvxopt.matrix(-np.ones((num_data,1)))
        G = cvxopt.matrix(-np.eye(num_data))
        A = cvxopt.matrix(labels,(1,num_data))
        b = cvxopt.matrix(np.zeros(1))
        h = cvxopt.matrix(np.zeros(num_data))

        alphas = np.ravel(cvxopt.solvers.qp(P, q, G, h, A, b)['x'])
        is_sv = alphas>1e-5
        self._support_vectors = data[is_sv]
        self._n_support = np.sum(is_sv)
        self._alphas = alphas[is_sv]
        self._support_labels = labels[is_sv]
        self._indices = np.arange(num_data)[is_sv]
        self.intercept = 0
        
        for i in range(self._n_support):
            self.intercept += self._support_labels[i] 
            self.intercept -= np.sum(self._alphas*self._support_labels*K[self._indices[i],is_sv].T)
        self.intercept /= self._alphas.shape[0]
        w = data.multiply(labels.reshape(num_data,1)).multiply(alphas.reshape(num_data,1)).sum(0)
        self.weights = csr_matrix(w) if self.kernel == "linear" else None

    def project(self,X):
        #print(X.shape)
        #print(self.weights.shape)
        if self.kernel=="linear":
            score = X.dot(self.weights.T) + self.intercept
        else:
            score = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                s = 0
                for alpha,label,sv in zip(self._alphas,self._support_labels,self._support_vectors):
                    s += alpha*label*self.kf[self.kernel](X[i],sv)
                score[i] = s
            score = score + self.intercept
        return score

    def predict(self,X):
        score = self.project(X)
        pred = score.sign().T.astype(int).toarray().ravel()
        return pred