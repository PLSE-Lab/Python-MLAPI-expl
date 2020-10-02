import itertools
import operator
import numpy as np

a = [1,2,4,8,16,32,64]
f_score = list()
def iterated(f_score,n):
    for i in range(1,len(a)+1):
        b = list(map(sum,(list(itertools.combinations(a,i)))))
        c = list(filter(lambda x : x < n, b))
        score = list(map(lambda x : n-x,c))
        f_score += score;
        #return f_score

iterated(f_score,21)
c = np.mean(f_score)
med = np.median(f_score)
print(c, med)
print(f_score)
del f_score[0:]
iterated(f_score,1000)
c = np.mean(f_score)
med = np.median(f_score)
print(c, med)
print(f_score)