import numpy as np


def polyfitLS(x, t, M):
    X = np.zeros((len(x),M+1))
    for i in range(len(x)):
        for j in range(M+1):
            X[i,j] = x[i]**j
    X = np.matrix(X).reshape(len(x),M+1)
    w_star = np.linalg.inv(X.H.dot(X)).dot(X.H).dot(t).T
    #print('w*',w_star.shape)

    t = np.matrix(t).reshape(len(t),1)
    #print('t',t.shape)
    #print('tH',t.H.shape)
    #print('X',X.shape)
    #print('XH',X.H.shape)
    X_w_star = X.dot(w_star)
    X_w_star_minus_t = X_w_star - t
    E = X_w_star_minus_t.H.dot(X_w_star_minus_t)
    #E = w_star.H.dot(X.H.dot(X.dot(w_star))) - w_star.H.dot(X.H.dot(t)) -\
    #        t.H.dot(X.dot(w_star)) + t.H.dot(t)

    return w_star, E

def polyfitRegLS(x, t, M, lam):
    X = np.zeros((len(x),M+1))
    for i in range(len(x)):
        for j in range(M+1):
            X[i,j] = x[i]**j
    X = np.matrix(X).reshape(len(x),M+1)

    w_star = np.linalg.inv(X.H.dot(X)+(np.identity(M+1)*lam*len(x)))\
             .dot(X.H).dot(t).T

    X_w_star = X.dot(w_star)
    X_w_star_minus_t = X_w_star - t

    return w_star

