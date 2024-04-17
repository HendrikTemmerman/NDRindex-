import numpy as np

def dis(A, B):
    return np.sqrt(np.sum((A - B)**2))

def NDRindex(A):
    n, m = A.shape
    aa = A.tolist()
    ans = 0
    tempdis = []

    for i in range(n - 1):
        for j in range(i + 1, n):
            tempdis.append(dis(aa[i], aa[j]))

    tempdis.sort()
    tempdiscnt = len(tempdis)

    if tempdiscnt % 4 == 0:
        S = tempdis[tempdiscnt // 4]
    else:
        S = (tempdis[tempdiscnt // 4] + tempdis[(tempdiscnt + (4 - tempdiscnt % 4)) // 4]) / 2

    S /= np.log10(n)
    oo = 0

    for pp in range(1, min(n, 100) + 1):
        a = aa.copy()
        t = 1
        sd = pp % n
        p = np.array(a[sd])
        q = [a[sd]]
        del a[sd]
        sum_val = 1
        sumo = 0
        inf = 1e9

        while True:
            o = inf
            for i, ai in enumerate(a):
                z = dis(p, ai)
                if z < o:
                    o = z
                    id = i

            if o == inf:
                break

            if o < S:
                q.append(a[id])
                t += 1
                p = np.zeros(m)

                for qi in q:
                    p += np.array(qi)

                p /= t
                del a[id]
            else:
                are = sum(dis(qi, p) for qi in q) / t

                if t > 3:
                    sumo += are
                    sum_val += 1
                    icenter.append(p)

                q = [a[id]]
                t = 1
                p = np.array(a[id])

        oo += (1.0 - (1.0 * sumo) / (S * sum_val))

    return oo / 100
