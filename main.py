import numpy as np

points = [[0, 0], [0.5, 0], [1, 0], [0, 0.5], [1, 0.5], [0, 1], [0.5, 1], [1, 1]]
def randomWalk(p, q0, N):
    np.random.seed(1246127846)
    q=[]
    q.append(q0)
    print(q)

    for n in range(0, N):

        i = np.random.randint(0, 8)
        q.append(np.divide(np.add(q[n],p[i]),3))
        if n%(N/100)==0:
            print(n/(N/100))

    return q


print(randomWalk(points,points[0],100))
