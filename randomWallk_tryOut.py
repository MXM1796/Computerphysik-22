import random



def random_walk(n):
    'Return coordinates after "n" steps of random walk'
    'Abbreviation for variable definition'
    x , y = 0, 0
    'Loop adding and subtracting infinitesimal distance'
    for i in range(n):
        (dx, dy) = random.choice([(0,1), (1,0),(-1,0), (0,-1)])
        'Abbreviation for adding and subtracting infinitesimal distance'
        x +=  dx
        y +=  dy
    return (x,y)


N = 1000

for i in range(25):
    walk = random_walk(10)
    print(walk, "Distance from home = ",
          abs(walk[0] + walk[1]))


