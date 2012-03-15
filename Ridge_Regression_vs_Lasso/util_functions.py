import matplotlib.pyplot as pl
import numpy.linalg as la
import numpy as np
import types

#################################################################

def ridge(X_init, y_init, d2_vec):
    X, y, _ = standardize(X_init, y_init)
    theta = np.empty((len(d2_vec), np.shape(X)[1]))

    for j,d2 in enumerate(d2_vec):
        d2I = np.diag(d2 * np.ones( np.shape(X)[1] ))
        XTX_plus_d2I = np.dot(X.T, X) + d2I
        theta[j] = np.dot( np.dot( la.inv(XTX_plus_d2I), X.T ), y )

    return theta

#################################################################

def standardize(X, y, opts=None):

    if opts == None:
        opts = {}
        opts['xmean'] = np.mean(X, axis=0)
        opts['ymean'] = np.mean(y)
        opts['scale'] = np.std(X-opts['xmean'], axis=0)

    zeros = np.empty(shape=np.shape(opts['scale']), dtype=bool)
    zeros = opts['scale'] == 0
    nonzeros = np.logical_not(zeros)

    X_s = np.empty(np.shape(X))
    X_s[:,nonzeros] = \
        (X[:,nonzeros] \
        - opts['xmean'][nonzeros]) \
        / opts['scale'][nonzeros]
    X_s[:,zeros] = 0
    y_s = y - opts['ymean']
    return X_s, y_s, opts

#################################################################

def deg_of_frdm(X_init, y_init, d2_vec):

    ## need to standardize??
    X, y, _ = standardize(X_init, y_init)
    U, S, Vt = la.svd(X)

    df = np.empty(len(d2_vec))
    for j,d2 in enumerate(d2_vec):
        df[j] = sum( [ s**2 / (s**2 + d2) for s in S])
    return df

#################################################################

def sqrd_err(X, y, k, function, d2_vec):

    n = np.shape(X)[0]
    d = np.shape(X)[1]

        # want n_round <-- the closest multiple of k 
        #   that is >= n
        #   e.g. (n=27, k=10) ==> n_round <-- 30
    n_round = n + (k - (n % k))
    errors = np.empty((k, n_round))

    for i in xrange(k):
        test = np.zeros(n, dtype=bool)
        test[0:i*k] = True
        start_test_again = min((i+1)*k, n-1)
        test[start_test_again:] = True

        train = np.logical_not(test)

        Xtrain, ytrain, opts = standardize(X[train], y[train])
        Xtest, ytest, _ = standardize(X[test], y[test], opts)
        theta = function(Xtrain, ytrain, d2_vec)

        for j in xrange(n_round):
            errors[i, j] = np.mean((ytest - np.dot(Xtest, theta[j]))**2)

    err_mean = np.mean(errors, axis=0)
    err_std = np.std(errors, axis=0)
    return err_mean, err_std

#################################################################

def lasso(X_init, y_init, d2_vec):

    THRESH = 1e-5
    X, y, opts = standardize(X_init, y_init)
    n = np.shape(X)[0]
    d = np.shape(X)[1]

    theta = ridge(X, y, d2_vec)
    theta_ = np.empty(np.shape(theta))

    for i,d2 in enumerate(d2_vec):
        theta_vec = theta[i]
        theta_vec_ = np.zeros(np.shape(theta_vec))

        while np.sum(np.abs(theta_vec - theta_vec_)) > THRESH:
            #print 'sum of errs = ', np.sum(np.abs(theta_vec - theta_vec_))
            theta_vec = theta_vec_
            theta_vec_ = theta_vec.copy()

            for j in range(d):
                a_j = 2*np.sum( X[:,j]**2 )
                c_j_vec = np.zeros((n,1))

                for k in range(n):
                    c_j_vec[k] = \
                        X[k,j] * (y[k] \
                        - np.dot(theta_vec_.T, X[k]) \
                        + theta_vec_[j] + X[k,j])
                c_j = np.sum(c_j_vec)

                if c_j < -d2:
                    theta_vec_[j] = (c_j + d2) / a_j
                elif c_j > d2:
                    theta_vec_[j] = (c_j - d2) / a_j
                else:
                    theta_vec_[j] = 0
        theta[i] = theta_vec_
    return theta

