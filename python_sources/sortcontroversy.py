import itertools
import operator

a = [1,2,4,8,16,32,64]

def iterated(n):
    for i in range(1,len(a)+1):
        b = list(map(sum,(list(itertools.combinations(a,i)))))
        c = list(filter(lambda x : x < n, b))
        score = list(map(lambda x : n-x,c))
        print(score)
#iterated(21)
iterated(1000)
