import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pplt
import sys

#################################################################
### Definitions

def parse_test_line(line):
    words = line.split()
    val_idx = 1
    param_vals = [float(val.split(':')[val_idx]) for val in words[1:]]
    return param_vals

def parse_data_line(line):
    words = line.split()
    val_idx = 1
    param_vals = [float(val.split(':')[val_idx]) for val in words[2:]]
    return (words[1], param_vals)

def print_mean_varnce(data):
    col_means = np.mean(data, axis=0)
    print 'initial col means:'
    print col_means
    for i,row in enumerate(data.T):
        row -= col_means[i]
        col_varnce = sum(row*row)/row.size
        print 'variance in col', i, '=', col_varnce

def plot2D(data, ratings):
    U, S, Vt = la.svd(data)
    Z_2 = np.dot(U[:,:2], np.eye(2)*S[:2])

    pluses = Z_2[(ratings > 0).squeeze()]
    minuses = Z_2[(ratings < 0).squeeze()]
    cent_p = np.mean(pluses, axis=0) 
    cent_m = np.mean(minuses, axis=0)

    ax = pplt.subplot(111)
    ax.scatter(pluses[:,0], pluses[:,1], s=20, c='r', marker='o')
    ax.scatter(cent_p[0], cent_p[1], s=70, c='r', marker='o')
    ax.scatter(minuses[:,0], minuses[:,1], s=20, c='g', marker='o')
    ax.scatter(cent_m[0], cent_m[1], s=70, c='g', marker='o')
    pplt.show()

def optimal_truncation(U,S,Vt):
    #TODO make a function to optimize the truncation point, given data
    k = 10
    return U[:,:k], S[:k], Vt[:k,:]

def eucl_dist(vec1, vec2):
    return np.sqrt(sum([(v1-v2)**2 for (v1,v2) in zip(vec1,vec2)]))

#################################################################
### Script

# grab problem parameters
n_data, n_params = [int(float(x)) for x in raw_input().split()]

# initialize arrays
data = np.empty((n_data, n_params))
ratings = np.empty((n_data,1))

# read training data up to n_data - 1
#for i,datum in enumerate(sys.stdin):
for i in range(n_data):
    datum = raw_input()
    rating, param_vals = parse_data_line(datum)
    ratings[i] = rating
    for j,param_val in enumerate(param_vals):
        data[i,j] =  param_val

# normalize the columns
col_means = np.mean(data, axis=0)
for i,row in enumerate(data.T):
    row -= col_means[i]
    col_varnce = sum(row*row)/row.size
    row /= col_varnce

#plot2D(data, ratings)

# get SVD
U, S, Vt = la.svd(data)

# find Wt so we can do whitening
data_rot = np.dot(data, Vt.T)
lambda_ = np.var(data_rot, axis=0)
D_vec = np.ones(lambda_.shape) / np.sqrt(lambda_)
Vdata = np.array([Vt[i]*lam for i,lam in enumerate(D_vec)]).T
Wt = np.dot(Vdata, Vt)

# comput data_zca to whiten our learning data
data_zca = np.dot(data, Wt)

pluses = data_zca[(ratings > 0).squeeze()]
minuses = data_zca[(ratings < 0).squeeze()]

#U_k, S_k, Vt_k = optimal_truncation(U,S,Vt)
#delta_hat = np.dot(U_k, np.dot(np.eye(len(S_k))*S_k, Vt_k))
#pluses = delta_hat[(ratings > 0).squeeze()]
#minuses = delta_hat[(ratings < 0).squeeze()]

cent_p = np.mean(pluses, axis=1) 
cent_m = np.mean(minuses, axis=1)
print 'cent_p =', cent_p
print 'cent_m =', cent_m
    
n_test = int(float(raw_input()))
for i, test in enumerate(sys.stdin):
    test_line = parse_test_line(test)
    params = np.array(test_line)

    # whiten the test parameters
    params_zca = np.dot(params, Wt)
    dist_p = eucl_dist(params_zca, cent_p)
    dist_m = eucl_dist(params_zca, cent_m)
    print dist_p, dist_m
    if dist_p > dist_m:
        print '-1'
    else:
        print '+1'
