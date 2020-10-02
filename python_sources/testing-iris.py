import random
import matplotlib.pyplot as plt
#create a d dimensional random vector inside a d dimensional hypercube
def randVect(d):
    v = []
    for i in range(d):
        v.append(random.random() - .5)
    return v
    
def dist(v1,v2):
    sum = 0
    for i in range(len(v1)):
        sum += (v1[i]-v2[i])**2
    
    return sum**.5
    
d = 100
n = 30
vectors = []
#30 random vectors
for i in range(n):
    vectors.append(randVect(d))

zero = []
#zero vector
for i in range(d):
    zero.append(0)

distToOrig = []
for i in range(len(vectors)):
    distToOrig.append(dist(zero, vectors[i]))
    
print(distToOrig)
plt.plot(distToOrig)
plt.ylabel("distance")
plt.xlabel("point")
plt.show()