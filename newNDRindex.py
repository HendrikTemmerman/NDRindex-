import numpy as np
from numpy.random import choice
from sklearn.metrics import adjusted_rand_score
from scipy.spatial import distance
import numpy as np

count = 0
def NDRindex(data,true_y):
    #n the sample number of the database
    n, d = data.shape
    A = choice(range(n))
    K = 1
    clen = 1

    # Assign each sample a cluster
    Y = np.full(n, -1)
    Y[A] = K
    gcenter = {}
    gcenter[K] = data[A]
    R = 0

    #the lower quartile distance of all point pairs and represents the range of data distribution
    M = calculate_range_from_M(data,n)


    while np.any(Y == -1):
        B_index = np.argmax(Y == -1)
        B = data[B_index]

        for j in range(n):
            point = data[j]
            if Y[j] == -1 and distance.euclidean(gcenter[K], point) < distance.euclidean(gcenter[K], B):
                B = point
                B_index = j

        if distance.euclidean(gcenter[K], B) < (M / np.log10(n)):
            Y[B_index] = K
            clen += 1
            for i in range(d):
                gcenter[K][i] = (gcenter[K][i] + B[i]) / clen
        else:
            tempsum = 0
            for j in range(n):
                point = data[j]
                if Y[j] == K:
                    tempsum += distance.euclidean(gcenter[K], point)
            tempsum = tempsum / clen
            R += tempsum
            K += 1
            clen = 1
            Y[B_index] = K
            gcenter[K] = B
    R = R/K
    NDRscore = 1 - (R / (M / np.log10(n)))
    ARIScore = adjusted_rand_score(Y, true_y)

    print('ARIscore: ', ARIScore)
    print('NDRscore: ', NDRscore)
    return NDRscore


def calculate_range_from_M(M,n):
    #distances = distance.pdist(M, metric='euclidean')
    distances = [distance.euclidean(M[i], M[j]) for i in range(n) for j in range(n)]
    lower_quartile_distance = np.percentile(distances, 25)
    return lower_quartile_distance




