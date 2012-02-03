from pylab import *
from scipy import *
import cPickle as pickle
from math import acos, degrees
from numpy.linalg import svd


### lisData.pkl is a pre-indexed corpus ###
# TODO create webcrawler using Django to scrape Wikipedia
# and compile a subset of semi-indexed pages

def loadData():
    data = pickle.load(open("lsiData.pkl"))
    return data

######################################################################

def extractData(data):
    urls = data.keys()
    tags = set()
    for tagfreq in data.values():
        tags.update(t for t,_ in tagfreq)
    tags = sorted(list(tags))
    return urls, tags

######################################################################

def normalize(x):
    xhat = reduce(lambda a,b: a+b, x, 0) / len(x)
    sq_diffs = [(z[0] - z[1])**2 for z in zip(x, [xhat]*len(x))]
    sigma = reduce(lambda a,b: a+b, sq_diffs, 0) / len(x)
    return (x - [xhat]*len(x)) / sigma

######################################################################

def generateCounts(data, tags, urls):
    # initialize A to zeros
    A = zeros((len(tags), len(urls)))

    # create something to look up tag nums in
    tag_nums = dict()
    for i, tag in enumerate(tags):
        tag_nums.update([(tag, i)])

    # update the non-zero parts of A
    for i, url in enumerate(urls):
        for tag_freq in data[url]:
            A[tag_nums[tag_freq[0]], i] = tag_freq[1]
            
    # normalize A
    for i, row in enumerate(A[:]):
        A[i] = normalize(row)

    return A

######################################################################

def truncatedSVD(A,k):
    m, n = shape(A)
    U, S, Vt = svd(A)
    return U[:,:k], S[:k], Vt[:k]

######################################################################

def query(q, U_k, S_k, Vt_k, tags, urls):
    """
    e.g. r = query(array(['ai','cs']), U_k, S_k, Vt_k, tags, urls)
    """
    SigInv_Ut = dot(inv(eye(len(S_k))*S_k), U_k.T)    
    qhat = dot(SigInv_Ut, q)
    r = []
    for i, url in enumerate(urls):
        dist = arccos(dot(qhat, Vt_k[:,i]) / (norm(qhat) * norm(Vt_k[:,i])))
        r.append((url, dist))
    r.sort(key=lambda x:x[1])
    return r
    

######################################################################

def buildEngine(k):
    """ 
    e.g. U_k, S_k, Vt_k, tag, url = buildEngine(2)
    """
    data = loadData()
    urls, tags = extractData(data)
    A = generateCounts(data, tags, urls)
    U_k, S_k, Vt_k = truncatedSVD(A, k)
    return U_k, S_k, Vt_k, tags, urls


######################################################################

def word2array(words):
    q = zeros(shape(tags))
    wordset = set(words)
    for i, tag in enumerate(tags):
        if tag in wordset:
            q[i] = 1
    return q

######################################################################

U_k, S_k, Vt_k, tags, urls = buildEngine(30)
q = word2array(['australia'])
r = query(q, U_k, S_k, Vt_k, tags, urls)
for i in xrange(5):
    print r[i][0]

