# 自主采样
import numpy as np
def index_bootstrap(num_data, prob):
    '''
    :param num_data: the index matrix of input, int
    :param prob: the probability for one index sample to be chose, >0
    return: index of chose samples, bool

    example:
    a=np.array([[1,2,3,4],[0,0,0,0]]).T
    rand_p = np.random.rand(4)
    b=np.greater(rand_p,0.5)
    b is the output, and we can use a[b] to locate data
    '''

    rand_p = np.random.rand(num_data)

    out = np.greater(rand_p, 1-prob)

    return out
