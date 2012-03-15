import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import types
import a5_util as a5

X = np.loadtxt('prostate.data')
d2_vec = np.logspace(-5, 10, num=100)
X_ = X[:,:-1]
y_ = X[:,-1]

####### Part 1: Data Visualization #######

plt.figure(1)
for i  in range(np.shape(X)[1]):
    for j in range(np.shape(X)[1]):
        if i != j:
            plt.subplot(9,9, 9*i + j + 1)
            plt.xticks([])
            plt.yticks([])
            plt.scatter(X[:,i], X[:,j])
plt.grid()
plt.show()

###### Part 2.a: Regularization Path for Ridge  #######

df = a5.deg_of_frdm(X_, y_, d2_vec)
theta = a5.ridge(X_, y_, d2_vec)

plt.figure(2)
plt.plot(df, theta)
plt.grid()
plt.show()

####### Part 2.b Cross Validation for Ridge ########

err_mean, err_std = a5.sqrd_err(X_, y_, 10, a5.ridge, d2_vec)
plt.figure(3)
plt.errorbar(df, err_mean, yerr=err_std)
plt.grid()
plt.show()

###### 3.a Regularization Path for Lasso ######

theta = a5.lasso(X_, y_, d2_vec)

plt.figure(4)
plt.plot(df, theta)
plt.grid()
plt.show()

###### 3.b Cross Validation for Lasso ######

err_mean, err_std = a5.sqrd_err(X_, y_, 10, a5.lasso, d2_vec)
plt.figure(5)
plt.errorbar(df, err_mean, yerr=err_std)
plt.grid()
plt.show()
