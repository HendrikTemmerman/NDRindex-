import numpy as np
from numpy.random import choice
from scipy.spatial import distance

def NDRindex(data,x):
    n, d = data.shape

    A = choice(range(n))
    K = 1
    clen = 1

    Y = np.full(n, -1)  # Cluster assignments
    Y[A] = K

    gcenter = {}
    gcenter[K] = data[A]

    R = 0
    BN = 0
    M = np.median([distance.euclidean(data[i], data[j]) for i in range(n) for j in range(i + 1, n)])

    while np.any(Y == -1):
        B = data[np.argmax(Y == -1)]
        for j in range(n):
            point = data[j]
            if Y[j] == -1 and distance.euclidean(gcenter[K], point) < distance.euclidean(gcenter[K], B):
                B = point
                BN = j

        if distance.euclidean(gcenter[K], B) < M / np.log10(n):
            Y[BN] = K
            clen += 1
            for i in range(d):
                gcenter[K][i] = (gcenter[K][i] * (clen - 1) + B[i]) / clen
        else:
            tempsum = 0
            for j in range(n):
                point = data[j]
                if Y[j] == K:
                    tempsum += distance.euclidean(gcenter[K], point)
            tempsum /= clen
            R += tempsum
            K += 1
            clen = 1
            Y[BN] = K
            gcenter[K] = B

    R /= K
    NDRscore = 1 - (R / (M / np.log10(n)))

    print(NDRscore)
    return NDRscore




