


###################
# declare variables



train = []                                          # all the information
curr = [[0 for x in range(28)] for y in range(28)]  # the currently treated line
tres = 9                                            # treshold for gamma



#############
# import file



import csv
with open('../input/train.csv', newline='') as file:
    lines = csv.reader(file, delimiter=',')
    for line in lines:
        train.append(line)



##############
# put in array



n = 19

for i in range(0, 28*28-1):
    curr[int((i-i%28)/28)][i%28] = int(train[n][i+1])



##########
# find box



minX = 0; maxX = 27; minY = 0; maxY = 27            # inside box treshold

out = False
for j in range(0, 27):
    for i in range(0, 27):
        if curr[i][j] > tres:
            minX = max(minX, j-1)
            out = True
            break
    if out:
        break

out = False
for j in range(27, 0, -1):
    for i in range(0, 27):
        if curr[i][j] > tres:
            maxX = min(maxX, j+1)
            out = True
            break
    if out:
        break

out = False
for i in range(0, 27):
    for j in range(0, 27):
        if curr[i][j] > tres:
            minY = max(minY, i-1)
            out = True
            break
    if out:
        break

out = False
for i in range(27, 0, -1):
    for j in range(0, 27):
        if curr[i][j] > tres:
            maxY = min(maxY, i+1)
            out = True
            break
    if out:
        break



##########################
# get average stroke width



n = 0; s = 0; sw = 0; tsw = 0; asw = 0

for i in range(minY, maxY):
    for j in range(minX, maxX):
        if curr[i][j] > tres:
            
            n += 1
            s = -1
            sw = -1
            
            for k in range(i, maxY):
                if curr[k][j] > tres:
                    sw += 1
                else:
                    break
            for k in range(i, minY, -1):
                if curr[k][j] > tres:
                    sw += 1
                else:
                    break
            
            for k in range(j, maxX):
                if curr[i][k] > tres:
                    s += 1
                else:
                    break
            for k in range(j, minX, -1):
                if curr[i][k] > tres:
                    s += 1
                else:
                    break
            
            if s < sw:
                sw = s
            
            tsw = tsw + sw

asw = tsw / n
print('asw: {}'.format(asw))



###################
# follow the stroke



i = i



############
# next block



i = i


