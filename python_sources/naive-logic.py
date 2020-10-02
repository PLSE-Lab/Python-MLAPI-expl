import pandas as pd
from sklearn import svm, metrics

def main():
    # The competition datafiles are in the directory ../input
    # Read competition data files:
    train = pd.read_csv("../input/train.csv")
    test  = pd.read_csv("../input/test.csv")

    grids=[]
    labels=[]

    firstRow=1
    totalRows=0

    classifier = svm.SVC(gamma=0.001)

    for sample in train.iterrows():
        if firstRow==0 and totalRows<19000:
            print("line: "+str(sample[0]))
            sample_payload=(sample[1])[1:785]
            sample_label=(sample[1])[0]
            sample_normalized_payload=normalize(sample_payload)
            sample_grid=grid_analysis(sample_label,sample_normalized_payload)
            grids.append(sample_grid)
            labels.append(sample_label)
        firstRow=0
        
        totalRows=totalRows+1
        

    print(str(len(grids))+":"+str(len(labels)))
    classifier.fit(grids, labels)

    firstRow=1
    totalRows=0
    for sample in train.iterrows():
        if firstRow==0 and totalRows<20000:
            print("line: "+str(sample[0]))
            sample_payload=(sample[1])[1:785]
            sample_label=(sample[1])[0]
            sample_normalized_payload=normalize(sample_payload)
            sample_grid=grid_analysis(sample_label,sample_normalized_payload)
            grids.append(sample_grid)
            labels.append(sample_label)
        firstRow=0

        totalRows=totalRows+1

    predicted=classifier.predict(grids)
    
    print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(labels, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(labels, predicted))


    # Write to the log:
    print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
    print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
    # Any files you write to the current directory get shown as outputs
    

def normalize(payload):
    return payload;

def grid_analysis(label, payload):
    
    # print("payload ")
    
    lines = [
             [[0,7],[1,0],[28,7]], [[0,14],[1,0],[28,14]], [[0,21],[1,0],[28,21]], 
             [[7,0],[0,1],[7,28]], [[14,0],[0,1],[14,28]], [[21,0],[0,1],[21,28]] 
            ]
    
    result=[]
    

    
    for x in lines:
        startXY=x[0]
        stepXY=x[1]
        stopXY=x[2]
        
        xy=startXY

        #print(xy)
        
        changes=0
        pixelstatus=0
        oldpixelstatus=0
        
        while not (xy[0]==stopXY[0] and xy[1]==stopXY[1]):

            #print(">>> x: "+str(xy[0])+ " - y: "+str(xy[1])+ " payload size:" + str(len(payload)))    
            #print("index:"+str(xy[1]*28+xy[0])+" payload:"+str(len(payload)))
            pixel=payload[xy[1]*28+xy[0]-1]
            #print(pixel)
            if pixel > 40 and oldpixelstatus==0:
                pixelstatus=1
                changes+=1
            if pixel < 40 and oldpixelstatus==1:
                pixelstatus=0
                changes+=1
            oldpixelstatus=pixelstatus    

            xy[0]=xy[0]+stepXY[0]
            xy[1]=xy[1]+stepXY[1]
            
        result.append(changes)        
            
    return result


if __name__=="__main__":
    main()