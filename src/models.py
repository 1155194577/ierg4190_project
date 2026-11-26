import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class LinearClassifier:
    def __init__(self, C=1.0, n_classes=100):
        self.C = C
        self.n_classes = n_classes

    def _augment(self, X):
        return np.hstack([X, np.ones((len(X), 1), np.float32)])

    def fit(self, X, y):
        Y = np.zeros((len(y), self.n_classes), np.float32)
        Y[np.arange(len(y)), y] = 1.0
        Xb = self._augment(np.asarray(X, np.float32))
        A = Xb.T @ Xb
        A[np.diag_indices_from(A)] += 1.0 / self.C
        self.W = np.linalg.solve(A, Xb.T @ Y)
        return self

    def predict(self, X):
        return np.argmax(self._augment(np.asarray(X, np.float32)) @ self.W, axis=1)


class SVM:
    def __init__(self,C = 1.0): 
        ## C = regularization params. large C = less regularization, small C = more regularization
        self.C = C 
        self.svm = LinearSVC(C=self.C, max_iter=10000)
    def fit(self, X, y): 
        self.svm.fit(X, y) 
        return self
    
    def predict(self, X): 
        return self.svm.predict(X)
     