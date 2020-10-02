# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import time


def permute(xs, low=0):# generator of all permutations of xs
    if low + 1 >= len(xs):
        yield xs
    else:
        for p in permute(xs, low + 1):
            yield p        
        for i in range(low + 1, len(xs)):        
            xs[low], xs[i] = xs[i], xs[low]
            for p in permute(xs, low + 1):
                yield p        
            xs[low], xs[i] = xs[i], xs[low]


def get_dist(path):# non-looping adjusted distance calc
    x2 = x_all[path[1:]]
    x1 = x_all[path[:-1]]
    y2 = y_all[path[1:]]
    y1 = y_all[path[:-1]]
    z  = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
    # incr step from [9] to [10] if primes[9]=0
    l  = [i for i in range(9,path.shape[0]-1,10) if primes[path[i]] == 0 ]
    z[l] *= 1.1
    dist = z.sum()
    return dist


def get_dist_all(path):# non-looping adjusted distance calc - before summation
    x2 = x_all[path[1:]]
    x1 = x_all[path[:-1]]
    y2 = y_all[path[1:]]
    y1 = y_all[path[:-1]]
    z  = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
    # incr step from [9] to [10] if primes[9]=0
    l  = [i for i in range(9,path.shape[0]-1,10) if primes[path[i]] == 0 ]
    z[l] *= 1.1
    return z

    
# read raw data
start_time = time.time()
input_dir = os.path.join(os.pardir, 'input')
print('  Loading data...')
data     = pd.read_csv(os.path.join(input_dir, 'cities.csv'), nrows=None)
N        = data.shape[0]
x_all    = data.X.values
y_all    = data.Y.values
x_all_32 = x_all
y_all_32 = y_all
print('    Time elapsed %.1f sec'%(time.time()-start_time))


#generate primes
print('  Generating primes...')
primes=np.ones(N)
primes[0:2]=0
for i in range(2,int(np.sqrt(N))+1):
    if primes[i] == 1 :
        l = [i*x for x in range(2,int(N/i))]
        primes[l] = 0
print('    Time elapsed %.1f sec'%(time.time()-start_time))


# read result of concord solver running for 5 hours: 1,516,910
# 1,516,677.63
path = np.array(pd.read_csv(os.path.join(input_dir, 'start.csv'), nrows=None)).reshape(197770).astype(np.int32)
print('Starting distance: %d'%(get_dist(path)))


# try swapping 2 line segments: 2-opt. Takes 30 minutes. 1,516,909
time_printed = start_time
d_curr = get_dist(path)
print('  Do 2-opt...')
changes = 0
for i in range(N-4, 0, -1):
    if i%1000 == 0 and time.time() > time_printed + 60:
        time_printed = time.time()
        print('    i: %dK. Time: %.1f sec. Changes: %d. Dist: %d'%(i/1000, time.time()-start_time, changes, d_curr))
    ii1, ii2 = path[i], path[i+1]
    jj1, jj2 = path[i+2:N-2], path[i+3:N-1]
    x1, y1   = x_all_32[ii1], y_all_32[ii1]
    x3, y3   = x_all_32[ii2], y_all_32[ii2]    
    x2, y2   = x_all_32[jj1], y_all_32[jj1]
    x4, y4   = x_all_32[jj2], y_all_32[jj2]
    d0       = np.sqrt((x1 - x3) * (x1 - x3) + (y1 - y3) * (y1 - y3)) # i to i+1
    d1       = np.sqrt((x2 - x4) * (x2 - x4) + (y2 - y4) * (y2 - y4)) # j to j+1
    d2       = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)) # i to j
    d3       = np.sqrt((x4 - x3) * (x4 - x3) + (y4 - y3) * (y4 - y3)) # i+1 to j+1
    if primes[ii1] == 0 and i%10 == 9: # step from i: incr step from [9] to [10] if primes[9]=0
        d0  *= 1.1
        d2  *= 1.1
    d01      = d1 + d0
    d23      = d2 + d3
    if (d23<d01).sum() > 0: # attempted change
        j     = np.argmax(d01 - d23) + i + 2 # select largest reduction in length
        path1 = np.append(path[:i+1].copy(), np.flip(path[i+1:j+1].copy()))
        path1 = np.append(path1, path[j+1:].copy())
        d_new = get_dist(path1)
        if d_curr > d_new: # actual change
            changes += 1
            print('reduce by %.2f'%(d_curr-d_new))
            path     = path1
            d_curr   = d_new


# 2.5-opt: take a point, remove it, and insert it in a later place.
# takes 30 min to run for s<25K. 1,516,719
time_printed = start_time
time_printed1 = start_time
z   = get_dist_all(path)
zz  = np.append(0,z.cumsum())
z1  = get_dist_all(path[1:])
zz1 = np.append(0,z1.cumsum())
print('  Do 2.5-opt...')
changes = 0
for s in range(25000, 8, -1):# no changes for s>25K. Skip <=8 - they are handled by permutations.
    if time.time()-time_printed > 60:
        time_printed = time.time()
        print('s=%d. Time elapsed %.1f sec. Dist: %d Changes: %d'%(s, time.time()-start_time, get_dist(path), changes))
    #path[i-1], path[i+1]
    dx   = x_all_32[path[:N-2]] - x_all_32[path[2:N]]
    dy   = y_all_32[path[:N-2]] - y_all_32[path[2:N]]
    d1a1 = np.sqrt(dx * dx + dy * dy)
    l1   = [i for i in range(9,dx.shape[0],10) if primes[path[i]] == 0 ]
    d1a1[l1] *= 1.1
    #path[i+s], path[i]
    dx   = x_all_32[path[s+1:N]] - x_all_32[path[1:N-s]]
    dy   = y_all_32[path[s+1:N]] - y_all_32[path[1:N-s]]
    d1a2 = np.sqrt(dx * dx + dy * dy)
    l2   = [i for i in range((200000+9-s)%10,dx.shape[0],10) if primes[path[i+1+s]] == 0 ]
    d1a2[l2] *= 1.1
    #path[i], path[i+s+1]
    dx   = x_all_32[path[1:N-1-s]] - x_all_32[path[2+s:N]]
    dy   = y_all_32[path[1:N-1-s]] - y_all_32[path[2+s:N]]
    d1a3 = np.sqrt(dx * dx + dy * dy)
    l3   = [i for i in range((200000+9-s-1)%10,dx.shape[0],10) if primes[path[i+1]] == 0 ]
    d1a3[l3] *= 1.1
    d1a  = d1a1[:d1a3.shape[0]] + d1a2[:d1a3.shape[0]] + d1a3 - zz1[1:1+d1a3.shape[0]] + zz1[s:s+d1a3.shape[0]]
    d0a  = zz[s+2:s+2+d1a3.shape[0]] - zz[:d1a3.shape[0]]
    if (d1a<d0a).sum() == 0 : # check all
       continue
    for i in range(1,N-1-s): # take point i, insert it after point i+s, then shift left by 1
        d0 = d0a[i-1]
        d1 = d1a[i-1]
        if d1 < d0:
            red1 = d0 - d1 # expected reduction in length, positive
            if time.time() > time_printed1 + 10:
                print('  move %d point %d steps further; length:%.3f->%.3f(-%.3f) Dist:%d'%(i,s,d0,d1,d0-d1,get_dist(path)))
                time_printed1 = time.time()
            p1 = np.append(path[i-1],path[i+1:i+s+1].copy())
            p1 = np.append(p1,path[i])
            p1 = np.append(p1,path[i+s+1])
            path1 = path.copy()
            path1[i-1:i+s+2] = p1
            d10 = get_dist(path)
            d11 = get_dist(path1)
            red2 = d10 - d11 # actual reduction in length, positive
            changes += 1
            path = path1.copy()
            # need to refresh this when path changes
            z   = get_dist_all(path)
            zz  = np.append(0,z.cumsum())
            z1  = get_dist_all(path[1:])
            zz1 = np.append(0,z1.cumsum())
            #path[i-1], path[i+1]
            dx   = x_all_32[path[:N-2]] - x_all_32[path[2:N]]
            dy   = y_all_32[path[:N-2]] - y_all_32[path[2:N]]
            d1a1 = np.sqrt(dx * dx + dy * dy)
            l1   = [i for i in range(9,dx.shape[0],10) if primes[path[i]] == 0 ]
            d1a1[l1] *= 1.1
            #path[i+s], path[i]
            dx   = x_all_32[path[s+1:N]] - x_all_32[path[1:N-s]]
            dy   = y_all_32[path[s+1:N]] - y_all_32[path[1:N-s]]
            d1a2 = np.sqrt(dx * dx + dy * dy)
            l2   = [i for i in range((200000+9-s)%10,dx.shape[0],10) if primes[path[i+1+s]] == 0 ]
            d1a2[l2] *= 1.1
            #path[i], path[i+s+1]
            dx   = x_all_32[path[1:N-1-s]] - x_all_32[path[2+s:N]]
            dy   = y_all_32[path[1:N-1-s]] - y_all_32[path[2+s:N]]
            d1a3 = np.sqrt(dx * dx + dy * dy)
            l3   = [i for i in range((200000+9-s-1)%10,dx.shape[0],10) if primes[path[i+1]] == 0 ]
            d1a3[l3] *= 1.1
            d1a  = d1a1[:d1a3.shape[0]] + d1a2[:d1a3.shape[0]] + d1a3 - zz1[1:1+d1a3.shape[0]] + zz1[s:s+d1a3.shape[0]]
            d0a  = zz[s+2:s+2+d1a3.shape[0]] - zz[:d1a3.shape[0]]
            

# try all permutations of subset of path
# length=8 takes 2.5 hours to run. 1,516,685
# length=9 takes 17  hours to run. 1,516,677
print('start permutations. Time elapsed %.1f sec'%(time.time()-start_time))
length = 9
subset = [1+i for i in range(length)]
ss = 1
for i in range(1,length+1):
    ss *= i
ll = np.zeros((ss, length+2), dtype=np.int32)
ll[:,0] = 0 # leading
ll[:,length+1] = length+1 # trailing
nn = 0
for p in permute(subset):
    ll[nn,1:length+1] = np.array(p) # much faster with np array than with list
    nn += 1
ll0 = ll.reshape(ll.shape[0]*ll.shape[1]).astype(np.int32)
time_printed = start_time
time_printed1 = start_time
time_saved = start_time
x_all_32 = x_all.astype(np.float32)
y_all_32 = y_all.astype(np.float32)
changes = 0
for offset in range(0,N-length-2): # loop over all offsets
    if offset%100 ==0 and time.time()-time_printed > 60:
        time_printed = time.time()
        print('  Offset %d. Time elapsed %.1f sec Dist: %df. Changes: %d'%(offset, time.time()-start_time, get_dist(path), changes))
    ll = path[ll0 + offset]
    x  = x_all_32[ll].copy().reshape(-1,length+2,)
    x1 = x[:,:length+1]
    x2 = x[:,1:length+2]
    dx = x1 - x2
    y  = y_all_32[ll].copy().reshape(-1,length+2)
    y1 = y[:,:length+1]
    y2 = y[:,1:length+2]
    dy = y1 - y2
    d  = np.sqrt(dx * dx + dy * dy)
    l_adj = [i for i in range((20009-offset)%10,length+1,10)] # list of column indices to be adjusted: every 10th step
    for j in l_adj:
        l = [i for i in range(nn) if primes[ll[i * (length + 2) + j]] == 0 ] # list of row indices to be adjusted: not from primes
        d[l,j] *= 1.1
    ds = d.sum(axis=1)
    d_min = ds.min()
    if d_min < ds[0]-0.0001: # found new best
        d_diff = ds[0] - d_min
        if time.time()-time_printed1 > 10:
            time_printed1 = time.time()
            print('found new best! Offset=%d, reduction from %.2f to %.2f - by %.2f. Dist:%d'%(offset, ds[0], d_min, d_diff, get_dist(path)))
        i_min = ds.argmin()
        path1 = path.copy()
        path1[offset:offset+length+2] = (ll.reshape(-1,length+2))[i_min,:].copy()
        path = path1.copy()
        changes += 1            



# Write submission file
if path[-1] > 0:
    path = np.append(path, 0) # return home
print(int(get_dist(path)))
out_df = pd.DataFrame({'Path': path})
out_df.to_csv('submission.csv', index=False)
print('    Time elapsed %.1f sec'%(time.time()-start_time))
