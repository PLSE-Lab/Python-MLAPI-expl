import numpy as np
import operator 

dataSet = np.array([[1,2,3],[4,5,6],[7,8,9]])

def autoNorm(dataSet):
    minCol = dataSet.min(0)
    maxCol = dataSet.max(0)
    diffMaxMin = maxCol - minCol
    numData = dataSet.shape[0]
    #diffDataMin = dataSet - np.tile(minCol,(numData,1))
    diffDataMin = dataSet - np.matrix(minCol)
    #diffMaxMin = np.tile(diffMaxMin,(numData,1))
    normData = diffDataMin/np.matrix(diffMaxMin)
    return normData
normData = autoNorm(dataSet)
print(normData)

'''
def createDataSet():
    train_x = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    train_y = ['a','a','b','b']
    return train_x, train_y
    
train_x,train_y = createDataSet()

def KnnClassify(test_x,train_x,train_y,k):
    train_size = train_x.shape[0]
    diff = np.tile(test_x, (train_size,1)) - train_x
    square_diff = diff**2
    sum_diff = square_diff.sum(axis=1)
    diff_res = sum_diff**0.5
    indexDisSort = diff_res.argsort()
    print (indexDisSort)
    count = {}
    for loop in range(k):
        key = train_y[indexDisSort[loop]]
        count[key] = count.get(key,0) + 1
    sortCount = sorted(count.items(),key=operator.itemgetter(1),reverse=True)
    return sortCount[0][0]

test_x =  np.array([0.1,0.1]) 
res = KnnClassify(test_x,train_x,train_y,2)
print(res)

def file2matrix(filename):
    fr = open(filename)
    lines = fr.readlines()
    numOfLines = len(lines)
    train_x = zeros((numOflines,3))
    train_y = []
    index_row = 0
    for loopline in lines:
        loopline = loopline.strip()
        loopline = loopline.split('\t')
        train_x[index_row,:] = loopline[0:3]
        index += 1
        train_y = train_y.append(int(loopline[-1]))
    return train_x,train_y
'''    

    
    
    
    
    
    
    
    
    
    
    
    
    