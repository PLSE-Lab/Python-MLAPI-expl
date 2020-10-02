import pandas as pd
import numpy as np # linear algebra
import math as mp


# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
#print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
#print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs
nArray = train.values

#sortedArray = sorted(nArray, key=sum)
# print(len(sortedArray))

def binarySearch( key):
    lowerBound = 0
    upperBound = len(sortedList)
    middle = int(( lowerBound + upperBound ) / 2)
    while lowerBound < upperBound and middle != lowerBound and middle != upperBound:
        if key < mp.sqrt(sum((avgSum[0]-np.delete(sortedList[middle][:],0))**2)/42000):
            upperBound = middle
        else:
            lowerBound = middle + 1
        middle = int(( lowerBound + upperBound ) / 2)
    return ( lowerBound + upperBound ) / 2

indexToCheck = 40000

avgSum = [[0]*28*28]
# count = 0
# for array in nArray:
#     img = np.delete(array[:],0)
#     avgSum += img
#     count += 1

# for i in range(10):
#     avgSum = avgSum / count
    
# print( avgSum )
# print( count )

def specialSort(array):
    array = np.delete(array[:],0)
    return mp.sqrt(sum((avgSum[0]-array)**2)/42000)
    # return mp.sqrt(sum((avgSum[0]-array)**2))
    
sortedList = sorted(nArray, key=specialSort)
 
    
# array = np.delete(nArray[indexToCheck][:],0)
# preValue = mp.sqrt(sum((avgSum[0]-array)**2)/42000)
# print(preValue)
# value = binarySearch(preValue-2)
# print()

# print( value )
# print( sortedList[int(value)][0] )
# print( nArray[indexToCheck][0] )
# print( mp.sqrt(sum((avgSum[0]-np.delete(nArray[indexToCheck][:],0))**2)/42000) )
# print( mp.sqrt(sum((avgSum[0]-np.delete(sortedList[int(value)][:],0))**2)/42000) )


# index = value-1
# distance = 100000000000000000000000000000000000
# indexHold = 0
# indexStart = index
# indexEnd = 0

# print()
# print("values")
# print(mp.sqrt(sum((avgSum[0]-np.delete(sortedList[int(index)][:],0))**2)/42000))
# print(int(preValue-1))
# print(int(preValue+2))
# while int(mp.sqrt(sum((avgSum[0]-np.delete(sortedList[int(index)][:],0))**2)/42000)) in range(int(preValue)-2,int(preValue)+2):
#     index+=1
#     holder = np.delete(sortedList[int(index)][:],0)
#     distTemp = mp.sqrt(sum((np.delete(nArray[indexToCheck][:],0)-holder)**2))
#     if distTemp < distance and distTemp != 0:
#         distance = distTemp
#         indexHold = index

# indexEnd = index

# print("endTest")
# index = 0
# indexHold2 = 0
# distance2 = 100000000000000000000000000000000000
# print(len(sortedList))
# for checker in sortedList:
    # index+=1
    # holder = np.delete(checker,0)
    # distTemp = mp.sqrt(sum((holder-np.delete(nArray[indexToCheck],0))**2))
    # if distTemp < distance2 and distTemp != 0:
    #     # print()
    #     # print(distTemp)
    #     # print(index)
    #     distance2 = distTemp
    #     indexHold2 = index
    
# print()
# print( indexHold2 )
# print( indexHold )
# print(distance2)
# print(distance)

# print(mp.sqrt(sum((avgSum[0]-np.delete(sortedList[int(indexHold2)][:],0))**2)/42000))

# print(mp.sqrt(sum((avgSum[0]-np.delete(sortedList[int(indexHold)][:],0))**2)/42000))

# print()

# print(mp.sqrt(sum((avgSum[0]-np.delete(sortedList[int(indexStart)][:],0))**2)/42000))

# print(mp.sqrt(sum((avgSum[0]-np.delete(sortedList[int(indexEnd)][:],0))**2)/42000))



print()
print()
print()
print("real Test")
total = 0
correct = 0

for checker in sortedList:
    if total == 101:
        break
    fixedChecker = np.delete(checker[:],0)
    preValue = mp.sqrt(sum((avgSum[0]-fixedChecker)**2)/42000)
    # preValue = mp.sqrt(sum((avgSum[0]-fixedChecker)**2))
    value = binarySearch(preValue-2)
    index = int(value)
    while (int(preValue)-2 <= int(mp.sqrt(sum((avgSum[0]-np.delete(sortedList[int(index)][:],0))**2)/42000))) != True:
        index+=1
    
    guessArray = [[100000000000000000000000000000000000,0],[100000000000000000000000000000000000,0],[100000000000000000000000000000000000,0]]
    while int(mp.sqrt(sum((avgSum[0]-np.delete(sortedList[int(index)][:],0))**2)/42000)) in range(int(preValue)-2,int(preValue)+2):
        holder = np.delete(sortedList[int(index)][:],0)
        distTemp = mp.sqrt(sum((fixedChecker-holder)**2))
        
        indexHolder = 0
        while indexHolder in range(3):
            if guessArray[indexHolder][0] > distTemp and distTemp != 0:
                guessArray[len(guessArray)-1][0] = distTemp
                guessArray[len(guessArray)-1][1] = sortedList[int(index)][0]
                guessArray = sorted(guessArray)
                break
            indexHolder+=1
        index+=1
        if index > len(sortedList) :
            break
    
    weights = [0] * 10
    for block in guessArray:
        weights[block[1]] += 1/block[0]
        
    guess = weights.index(max(weights))
    
    if checker[0] == guess:
        correct += 1
    
    total += 1
    print(total)
    
print( correct )
print( total )
print( correct/ total )
















# array = np.delete(nArray[indexToCheck],0)
# value = binarySearch(mp.sqrt(sum((avgSum[0]-array)**2)))

# print(value)
# print(sortedList[int(value)][0])
# print(sortedList[int(value)-1][0])
# print(sortedList[int(value)+1][0])
# print(nArray[indexToCheck][0])

# print()
# print(mp.sqrt(sum((avgSum[0]-np.delete(sortedList[int(value)],0))**2)))
# print(mp.sqrt(sum((avgSum[0]-np.delete(sortedList[int(value)-1],0))**2)))
# print(mp.sqrt(sum((avgSum[0]-np.delete(sortedList[int(value)+1],0))**2)))
# print(mp.sqrt(sum((avgSum[0]-np.delete(nArray[indexToCheck],0))**2)))

# print()
# print(mp.sqrt(sum((np.delete(sortedList[int(value)],0) - np.delete(nArray[indexToCheck],0))**2)))
# print(mp.sqrt(sum((np.delete(sortedList[int(value)-1],0) - np.delete(nArray[indexToCheck],0))**2)))
# print(mp.sqrt(sum((np.delete(sortedList[int(value)+1],0) - np.delete(nArray[indexToCheck],0))**2)))

# distance = 100000000000000000000000000000000000
# index = -1
# indexHold = 0
# for testing in sortedList:
#     index += 1
#     tempHold = mp.sqrt(sum((np.delete(nArray[indexToCheck],0)-np.delete(testing,0))**2))
#     tempHold = int(tempHold)
#     if tempHold < distance and tempHold > 0:
#         print(distance)
#         distance = tempHold
#         indexHold = index


# print()
# print(distance)
# print(indexHold)
# print(mp.sqrt(sum((np.delete(sortedList[indexHold],0) - np.delete(nArray[indexToCheck],0))**2)))
# print(mp.sqrt(sum((avgSum[0]-np.delete(nArray[indexToCheck],0))**2)))
# print(mp.sqrt(sum((avgSum[0]-np.delete(sortedList[indexHold],0))**2)))


















# test = sortedArray[max(0, min(5, len(sortedArray)-1))]
# test2 = np.delete(test,0)

# print( binarySearch(sum(test2)))

# indexHolder = 0
# index= 0
# distance = 100000000000000000000000000000000000

# sI = 0
# sD = 100000000000000000000000000000000000
# target = sum(test2**2)

# print(target )
# for array in sortedArray:
#     img = np.delete(array,0)
    
#     if abs(sum(test2**2) - sum(img**2)) != 0 and abs(sum(test2**2) - sum(img**2)) < sD:
#         sD = abs(sum(test2**2) - sum(img**2))
#         sI= index
    
#     sumHolder = sum((test2-img)**2)
#     if sumHolder < distance and sumHolder != 0:
#         distance = sumHolder
#         indexHolder =index
#     index+=1

# print()
# print( sum(sortedArray[indexHolder]**2))

# print()
# print (indexHolder)
# print (distance)

# print()
# print( sum((test2-np.delete(sortedArray[indexHolder],0))**2))
# print( sum((np.delete(sortedArray[indexHolder],0)-test2)**2))

# print()
# print(sD)
# print(sI)
# print(test[0])
# print(sortedArray[sI][0])
# print(sum(test2))
# print(sum(np.delete(sortedArray[sI],0)))
# print(sum(test2 **2))
# print(sum(np.delete(sortedArray[sI],0)**2))
# print(sum((test2-np.delete(sortedArray[sI],0))**2))












    
# correct = 0
# total = 0
# currentOuterLoop = 0
# for array in sortedArray:
#     img = np.delete(array,0)
#     index = binarySearch(sum(img))
#     lowerBound = max(0, min(index - 1000, len(sortedArray)-1))
#     upperBound = max(0, min(index + 1000, len(sortedArray)-1))
    
#     index = -1
#     i = lowerBound
#     # guess = [[0,0]]* int((upperBound - lowerBound + 1)) 
#     guess = [[0,0] for i in range(int((upperBound - lowerBound + 1)))]
#     # print(guess)
#     for i in range(int(lowerBound),int(upperBound+1)):
#         if currentOuterLoop == i:
#             continue
#         index+=1
#         # print(i)
#         # print(index)
#         # print(guess)
#         # print()
#         arrayTemp = np.delete(sortedArray[i],0)
#         # guess[index][0] = mp.sqrt(sum((img-arrayTemp)**2)) #.969
#         guess[index][0] = sum(abs((img-arrayTemp))) #.969
#         guess[index][1] = sortedArray[i][0]
        
#     # print(guess)
#     guess = sorted(guess)
#     # print(guess)
    
#     weights = [0 for i in range(10)]
#     # print( weights)
    
#     i = 0
#     for block in guess:
#         if block[0] != 0:
#             weights[block[1]] += 1/block[0]
#         i+=1
#         if i == 3:
#             break
#     # print( weights)
    
#     prediction = weights.index(max(weights))
    
#     currentOuterLoop += 1
    
#     if prediction == array[0]:
#         correct+=1
#     total+=1
#     if total == 1000:
#         break

# print()
# print(correct)
# print(total)
# print(correct/total)





































# current = 0
# currentI = 0
# sumTA = 0
# for array in nArray:
#     img = (np.delete(array,0))
#     guess = [0,0,0]
#     distance = [100000000000000000000000000000000000,100000000000000000000000000000000000,100000000000000000000000000000000000]
    
#     tempHold = [0,0]
#     currentJ = 0
#     for imgTrain in nArray:
#         if currentJ == currentI:
#             currentJ+=1
#             continue
#         testImg = (np.delete(imgTrain,0))
#         sumTemp2 = sum((img - testImg)**2)
#         sumTemp = sum(abs(img-testImg))
#         x = 0
#         if sumTemp < distance[x]:
#             distance[x] = sumTemp
#             guess[x] = imgTrain[0]
#             tempHold[0] = sum(testImg)
#         if sumTemp2 < distance[1]:
#             distance[1] = sumTemp2
#             guess[x] = img[0]
#             tempHold[1] = sum(testImg)
#         currentJ += 1
    
#     print(distance[0])
#     print(sum(img))
#     print(tempHold[0])
#     print(tempHold[1])
#     print("")
    
#     sumTA += distance[0]
    
    
#     current+=1
#     if current == 10:
#         break;
#     currentI += 1

# print(sumTA)
# print(sumTA/(current+1))


















#print(nArray[0][0])
#print(np.delete(nArray[0], 0))
#print(sum(abs(nArray[0] - nArray[1])))
#print(sum((nArray[0] - nArray[1])**2))
#print(nArray[0])

def normalizeImage(img):
    range = img.max()-img.min()
    minValue = img.min()
    return (img - minValue) * 255 / range

# trainSet = [[0] * 28*28,[0] * 28*28,[0] * 28*28,[0] * 28*28,[0] * 28*28,[0] * 28*28,[0] * 28*28,[0] * 28*28,[0] * 28*28,[0] * 28*28]
# trainSetCount = [0,0,0,0,0,0,0,0,0,0]
# for array in nArray:
#     img = normalizeImage(np.delete(array,0))
#     trainSet[array[0]] = trainSet[array[0]] + img
#     trainSetCount[array[0]] += 1
    
# i = 0
# for i in range(0,9):
#     trainSet[i] = trainSet[i] / trainSetCount[i]
    
#     correct = 0
# total = 0
# for array in nArray:
#     img = normalizeImage(np.delete(array,0))
#     guess = 0
#     lowest = 100000000000000000000000000000000000
#     currentI = 0
#     for imgTrain in trainSet:
#         sumTemp = sum((imgTrain - img)**2)
#         if sumTemp < lowest:
#             guess = currentI
#             lowest = sumTemp
#         currentI += 1
#     if guess == array[0]:
#         correct += 1
#     total += 1

# print( correct )
# print( total )
# print( correct / total )
    
#print(trainSet)
#print(trainSetCount)

def distance(imgA, imgB):
    return sum((imgA - imgB)**2)

# correct = 0
# total = 0
# currentJ = 0
# for array in nArray:
#     img = (np.delete(array,0))
#     guess = [0,0,0]
#     distance = [100000000000000000000000000000000000,100000000000000000000000000000000000,100000000000000000000000000000000000]
#     currentI = 0
#     for imgTrain in nArray:
#         if currentJ == currentI:
#             currentI+=1
#             continue
#         testImg = (np.delete(imgTrain,0))
#         # sumTemp = sum((img - testImg)**2)
#         sumTemp = sum(abs(img-testImg))
#         x = 0
#         for x in range(0,3):
#             if sumTemp < distance[x]:
#                 guessTemp = 0
#                 distanceTemp = 0
#                 for x in range(0,3):
                    
#                 distance[x] = sumTemp
#                 guess[x] = imgTrain[0]
#                 break
#         currentI+=1

#     guessI = -1
#     guessPower = 0
#     # print( guess )
#     # print( distance )
    
#     weights = [0,0,0,0,0,0,0,0,0,0]
    
#     x = 0
#     for x in range(0,3):
#         weights[guess[x]] += 1/distance[x]
        
#     # print (weights)
    
#     largestNumber = 0
#     largestWeight = 0
#     x = 0
#     for x in range(0,10):
#         if weights[x] > largestWeight:
#             largestWeight = weights[x]
#             largestNumber = x
#     # 
#     # print(largestNumber)
#     if largestNumber == array[0]:
#         correct+=1
#     total+=1
#     currentJ+=1
#     print(correct/total)
#     if currentJ == 100:
#         break

# print(correct)
# print(total)
# print(correct/total)














