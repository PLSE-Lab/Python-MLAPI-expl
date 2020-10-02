import numpy as np
import pandas as pd

class MyPAR():

    def __init__(self, C=0.01, eps=0.001, mode="PA"):
        self.w = None
        self.C = C
        self.eps = eps
        self.mode = mode


    def online_train(self, X, y):
        if (type(X) is pd.DataFrame):
            X = X.to_numpy()
        if (type(y) is pd.DataFrame):
            y = y.to_numpy().ravel()

        n_examples = X.shape[0]
        n_features = X.shape[1]
        if self.w is None:
            self.w = np.zeros(n_features + 1)

        for i in range(n_examples):
            y_pred = self.predict(X[i])
            l = self.loss(y_pred, y[i])
            t = self.tao(l, X[i])
            X_stretched = np.insert(X[i], 0, 1)
            self.w = self.w + np.sign(y[i] - y_pred) * t * X_stretched


    def predict(self, X):
        if (type(X) is pd.DataFrame):
            X = X.to_numpy()
        if self.w is not None:
            return X.dot(self.coef_) + self.intercept_


    @property
    def intercept_(self):
        if self.w is not None: 
            return self.w[0]


    @property
    def coef_(self):
        if self.w is not None:
            return self.w[1:]


    def loss(self, y_pred, y_true):
        loss = np.array(np.abs(y_pred - y_true) - self.eps)
        loss[loss <= 0] = 0
        return loss


    def PA(self, loss, X):
        X_norm = np.linalg.norm(X, axis=0)
        if X_norm == 0:
            return 0
        return loss / (X_norm ** 2)


    def PA1(self, loss, X):
        pa1 = np.array(self.PA(loss, X))
        pa1[pa1 > self.C] = self.C
        return pa1


    def PA2(self, loss, X):
        if self.C == 0:
            return 0
        X_norm = np.linalg.norm(X, axis=0)
        return loss / ((X_norm ** 2) + (1 / (2 * self.C)))


    def tao(self, loss, X):
        if (self.mode == "PA"):
            return self.PA(loss, X)
        elif (self.mode == "PA1"):
            return self.PA1(loss, X)
        elif (self.mode == "PA2"):
            return self.PA2(loss, X)


    def __repr__(self):
        hypothesis = "h"
        if self.coef_ is None:
            return hypothesis + "(X) = None"
        else:
            sb = hypothesis + "(X) = %.2f" % self.intercept_
            for i in range(len(self.coef_)):
                if self.coef_[i] >= 0:
                    sb += " + %.2f" % abs(self.coef_[i])
                else:
                    sb += " - %.2f" % abs(self.coef_[i])
                sb += "*X" + str(i+1)
            return sb



if __name__ == "__main__":
    from sklearn.linear_model import PassiveAggressiveRegressor

    X = np.array([[1,2,3], [4,5,6], [7,8,9], [-1,-2,-3], [12,11,10], [15,14,13], [0,0,0], [-10,-20,-30], [5,5,5]])
    y = np.array([14, 32, 50, -14, 64, 82, 0, -140, 30])
    ex = np.array([[1,1,1]])

    my_clf = MyPAR(C=0.1, eps=0.01, mode="PA1")
    my_clf.online_train(X, y)
    my_pred = my_clf.predict(ex)

    sk_clf = PassiveAggressiveRegressor(C=0.1, epsilon=0.01, shuffle=False, loss="epsilon_insensitive")
    sk_clf.partial_fit(X, y)
    sk_pred = sk_clf.predict(ex)

    print(my_clf)
    print(MyPAR.__repr__(sk_clf))
    print(sk_pred, my_pred)

