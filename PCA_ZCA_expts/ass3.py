import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as pl

## DEFINITIONS ###

def load_patches(images, n=10000, p=20):
    rand = np.random.RandomState(seed=0)
    X = np.empty((n, p**2))
    m, d, _ = images.shape
    for i in xrange(n):
        x, y = rand.randint(d-p, size=2)
        j = rand.randint(m)
        X[i,:] = images[j, y:y+p, x:x+p].flat
    return X

###  SCRIPT ###

# images are so-called 'natura; images'
images = np.load('images.npy')
pl.set_cmap(pl.cm.Greys_r)
X = load_patches(images)

U,S,Vt = la.svd(X, full_matrices=False)

# truncate the basis at k = 25
k = 25
Vt_k = Vt[0:k]

pl.figure()
for i, line in enumerate(Vt_k):
    pl.subplot(5,5,i+1)
    pl.imshow(line.reshape(20,20))

# find the top five bases for image 0, based on 
# magnitude of coeficients
coefs = [(i, U[0,i]*S[i]) for i in xrange(S.shape[0])]
coefs.sort(key = lambda x: abs(x[1]))
coefs.reverse()
top_coefs = [coef for coef in coefs[0:5]]
top_Vs = [(coef[1], Vt[coef[0]]) for coef in top_coefs]

# plot them
pl.figure()
pl.subplot(2,3,1)
pl.imshow(X[0].reshape((20,20)))
for i, aV in enumerate(top_Vs):
    pl.subplot(2,3,i+2)
    pl.title(aV[0])
    pl.imshow(aV[1].reshape((20,20)))

#######################

# 
Xrot = dot(X, Vt.T)
lambda_ = np.var(Xrot, axis=0)
Xwhit = Xrot / np.sqrt(lambda_)
Xzca = np.dot(Xwhit, Vt)

figure()
pl.subplot(1,2,1)
pl.title('un-whitened')
pl.imshow(X[10].reshape((20,20)))
pl.subplot(1,2,2)
pl.title('whitened')
pl.imshow(Xzca[10].reshape((20,20)))

### 1.4.ii ###

## problem is that the lambda vals haven't been
## sqrt'ed and taken under ones
D_vec = ones(shape(lambda_)) / sqrt(lambda_)
VX = np.array([Vt[i]*lam for i,lam in enumerate(D_vec)]).T
Wt = dot(VX, Vt)
figure()
for i,line in enumerate(Wt[0:25]):
    subplot(5,5,i+1)
    imshow(line.reshape((20,20)))

### 1.4.iii ###

Wt_inv = inv(Wt)
figure()
for i,line in enumerate(Wt_inv[0:25]):
    subplot(5,5,i+1)
    imshow(line.reshape((20,20)))
