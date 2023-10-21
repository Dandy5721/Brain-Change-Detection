import numpy as np


class Euclidean:
    def __init__(self):
        self.name = 'Euclidean'

    def logm(self, X):
        return X

    def expm(self, X):
        return X

    def logmm(self, X, Y):
        return Y - X

    def expmm(self, X, Y):
        return X + Y

    def norm(self, X):
        return np.linalg.norm(X, ord=2)

    def distance(self, X, Y):
        return self.norm(Y - X)


class SPD:
    def __init__(self):
        self.name = 'SPD'

    def logm(self, X):
        S, U = np.linalg.eigh(X)
        S = np.diag(np.log(S))
        return U.dot(S).dot(U.T)

    def expm(self, X):
        S, U = np.linalg.eigh(X)
        S = np.diag(np.exp(S))
        return U.dot(S).dot(U.T)

    def logmm(self, X, Y):
        return self.logm(Y) - self.logm(X)

    def expmm(self, X, Y):
        return self.expm(self.logm(X) + Y)

    def norm(self, X):
        return np.linalg.norm(X, ord='fro')

    def distance(self, X, Y):
        return self.norm(self.logm(Y) - self.logm(X))
