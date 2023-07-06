import random

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
def Model_Selection(centers, weights, biases, spreads, best_x, num_set):

    num_models = len(centers)
    results = np.zeros((num_models, 1))
    selected_index = []
    for i in range(0, num_models):
        c, w, b, s= centers[i], weights[i], biases[i], spreads[i]
        results[i] = rbf_predict(c, w, b, s, best_x)
    gap = int(num_models/num_set)
    results = np.asarray(results).flatten()
    sorted_index = np.argsort(results)

    #random select num_set models
    for j in range(0, num_set):
        selected_index.append(random.sample([k for k in sorted_index[j*gap:(j+1)*gap]],1)[0])
    # print(selected_index)
    return selected_index

def _dist(Mat1, Mat2):
    '''
    rewrite euclidean distance function in Matlab: dist
    :param Mat1: matrix 1, M x N
    :param Mat2: matrix 2, N x R
    output: Mat3. M x R
    '''
    Mat2 = Mat2.T

    return cdist(Mat1,Mat2)




class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""

    def __init__(self, k, kernel=None):
        self.k = k
        self.kernel = kernel
        self.Overlap = 1.0
        self.Centers = []



    def local_update(self, X, Y):
        SamIn = X.T
        SamOut = Y.T
        InDim, SamNum = SamIn.shape[0], SamIn.shape[1]
        k_means = KMeans(n_clusters=self.k).fit(X)
        self.Centers = k_means.cluster_centers_
        AllDistances = _dist( self.Centers,self.Centers.T)
        Maximum = AllDistances.max()
        for i in range(self.k):
            AllDistances[i, i] = Maximum+1
        AllDistances = np.where(AllDistances !=0, AllDistances, 0.000001)
        Spreads = self.Overlap*np.min(AllDistances, axis=0).reshape(-1,1)
        Distance = _dist(self.Centers, SamIn)
        SpreadsMat = np.tile(Spreads, (1, SamNum))
        #矩阵点除
        HiddenUnitOut = np.exp(-(Distance/SpreadsMat)**2)
        HiddenUnitOutEx = np.hstack((HiddenUnitOut.T,np.ones((SamNum,1)))).T
        W2Ex = np.dot(SamOut, np.linalg.pinv(HiddenUnitOutEx))
        # print(W2Ex.shape)
        W2 = W2Ex[:, 0:self.k]
        B2 = W2Ex[:, self.k]
        return self.Centers, W2, B2, Spreads



def rbf_predict(centers,weights,bias,spreads,test_x):
    N = test_x.shape[0]
    TestDistance = _dist(centers, test_x.T)
    TestSpreadMat = np.tile(spreads, (1, N))
    TestHiddenUintOut = np.exp(-(TestDistance / TestSpreadMat) ** 2)
    TestNNOut = np.dot(weights, TestHiddenUintOut) + bias
    TestNNOut = TestNNOut.T

    return TestNNOut