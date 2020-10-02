#!/usr/bin/env python
# coding: utf-8

# Here I will merge [my Something Different kernel ](https://www.kaggle.com/jimpsull/something-different) with [Alexander Firsov's](https://www.kaggle.com/alexfir) [fast-test-set-reading kernel](https://www.kaggle.com/alexfir/fast-test-set-reading)
# - There is an approach here that is different from much of what is in the publicly available kernels
# - I am not an astronomer.  A different perspective can sometimes be a good thing.  You may find that a few of these features can be combined with your features

# **Begin Alexander Firsov's code**

# The kernel is an attempt to overcome kernels limitation (RAM, disk space) and allow fast test_set reading from kernels. It also can be helpful for owners of low memory computers.
# 
# The idea is to store each column as numpy array, then access them using numpy.memmap and combine column into pandas when needed. Reading is available by object id or in chunks. A chunk may contain multiple object ids, object ids are not spread between chunks.
# 
# Data preparation is done in two kernels because of max disk space limitation. See [kernel 1](https://www.kaggle.com/alexfir/test-set-columns-part-1) and [kernel 2](https://www.kaggle.com/alexfir/test-set-columns-part-2). Output of the kernels is added as input to this kernel.

# In[ ]:


import numpy as np
import pandas as pd
import os.path
import time

COLUMN_TO_TYPE = {
    'object_id': np.int32,
    'mjd': np.float32,
    'passband': np.int8,
    'flux': np.float32,
    'flux_err': np.float32,
    'detected': np.int8
}

part1_directory = r'../input/test-set-columns-part-1'
part2_directory = r'../input/test-set-columns-part-2'

COLUMN_TO_FOLDER = {
    'object_id': part2_directory,
    'mjd': part2_directory,
    'passband': part2_directory,
    'flux': part1_directory,
    'flux_err': part1_directory,
    'detected': part1_directory
}


def init_reading():
    info = {}
    object_range_file_path = os.path.join(COLUMN_TO_FOLDER['object_id'], 'object_id_range.h5')
    print('reading {}'.format(object_range_file_path))
    object_id_to_range = pd.read_hdf(object_range_file_path, 'data')
    info['object_id_to_range'] = object_id_to_range
    id_to_range = object_id_to_range.set_index('object_id')
    info['object_id_start'] = id_to_range['start'].to_dict()
    info['object_id_end'] = id_to_range['end'].to_dict()

    records_number = object_id_to_range['end'].max()

    mmaps = {}
    for column, dtype in COLUMN_TO_TYPE.items():
        directory = COLUMN_TO_FOLDER[column]
        file_path = os.path.join(directory, 'test_set_{}.bin'.format(column))
        mmap = np.memmap(file_path, dtype=COLUMN_TO_TYPE[column], mode='r', shape=(records_number,))
        mmaps[column] = mmap

    info['mmaps'] = mmaps

    return info


def read_object_info(info, object_id, as_pandas=True, columns=None):
    start = info['object_id_start'][object_id]
    end = info['object_id_end'][object_id]

    data = read_object_by_index_range(info, start, end, as_pandas, columns)
    return data


def read_object_by_index_range(info, start, end, as_pandas=True, columns=None):
    data = {}
    for column, mmap in info['mmaps'].items():
        if columns is None or column in columns:
            data[column] = mmap[start: end]

    if as_pandas:
        data = pd.DataFrame(data)

    return data


def get_chunks(info, chunk_size=1000):
    object_id_to_range = info['object_id_to_range']
    end_of_file_offset = object_id_to_range['end'].max()
    start_offsets = object_id_to_range['start'].values[::chunk_size]
    end_offsets = object_id_to_range['end'].values[(chunk_size - 1)::chunk_size]

    end_offsets = list(end_offsets) + [end_of_file_offset]

    chunks = pd.DataFrame({'start': start_offsets, 'end': end_offsets})
    chunks = chunks.values.tolist()

    return chunks


# Before reading data, call init function and get info object that will be used later.

# In[ ]:


info = init_reading()


# Reading can be done just by object id, like in code below.

# In[ ]:


# single object read as pandas object, first object
object_info13 = read_object_info(info, 13)
object_info13.head()


# In[ ]:


# last object from test_set
object_info104853812 = read_object_info(info, 104853812)
object_info104853812.tail()


# Data can be returned as pandas object or as dict of numpy arrays (numpy memmap). 

# In[ ]:


object_info104853812 = read_object_info(info, 104853812, as_pandas=False, columns=['flux', 'flux_err'])
object_info104853812['flux'][-5:]


# Let's read all test set and perform some operation (we need doing something as data is not read until we use them) and measure required time.

# In[ ]:


object_ids = info['object_id_to_range']['object_id'].values.tolist()
start_time = time.time()
records_read = 0
for object_id in object_ids:
    object_info = read_object_info(info, object_id, columns=['flux'], as_pandas=False)
    flux = object_info['flux']
    records_read += flux.shape[0]
    max = flux.max()

print("Single field reading took {:6.4f} secs, records = {}".format((time.time() - start_time), records_read))


# Not bad, but still only reading took a lot of time, even so single field was read and no pandas object was created.
# Now let's see what is performance of chunk reading. Each chunk contains 10000 objects and pandas object is created.

# In[ ]:


start_time = time.time()
records_read = 0
chunks = get_chunks(info, 10_000)
for index_start, index_end in chunks:
    data = read_object_by_index_range(info, index_start, index_end)
    flux = data['flux']
    max = flux.max()
    records_read += data.shape[0]

print("Chunks reading took {:6.4f} secs, records = {}".format((time.time() - start_time), records_read))


# **Begin Jim Sullivan's code**

# In[ ]:


import warnings
from matplotlib import pyplot as plt
import copy
import scipy.stats as ss

print(os.getcwd())
print(os.listdir('../input/PLAsTiCC-2018'))
print('Getting Base (meta) dataFrame')
bdf=pd.read_csv('../input/PLAsTiCC-2018/test_set_metadata.csv')
print(bdf.shape)

#this is what Alex's code avoids having to do
#print('Getting raw (lightcurve) dataFrame')
#rdf=pd.read_csv('../input/training_set.csv')
#print(rdf.shape)


# **I'd like to replace getLCDFArchive with getLCDF based on Alex's code**

# In[ ]:



def getLCDF(objid, info, show=False, lepb=2, hepb=5):
    # single object read as pandas object, first object
    # Alex's code for getting a light curve data frame
    #original unfiltered light curve data frame
    ulcdf = read_object_info(info, objid)
    #unfiltered lcdf
    #this may not be necessary and may be causing my slowdown as size increases
    #ulcdf=copy.deepcopy(oulcdf)
    filterLow = ulcdf.loc[:,'passband']==lepb
    filterHigh = ulcdf.loc[:,'passband']==hepb
    filterPb = filterLow | filterHigh
    lcdf=ulcdf.loc[filterPb,:]
    if show:
        plt.plot(lcdf.loc[:,'mjd'],lcdf.loc[:,'flux'])
        plt.show()
        
    return lcdf

objA=bdf.loc[0,'object_id']
objB=bdf.loc[3000000, 'object_id']

alcdf=getLCDF(objA, info, show=True)
blcdf=getLCDF(objB, info, show=True)


# **I hope the hardest part is over**
# - I've now integrated my code with Alex's code.
# - I can extract lightcurves and extract features without loading the whole test set into one dataFrame
# - More comments and markdown on my original Feature Extractor Kernel

# In[ ]:


def divideLcdf(elcdf, ddf, lep=2, hep=5):
    #lcdf=copy.deepcopy(elcdf)
    #this simple date cutting works on the ddf objects
    if ddf:
        minDate=np.min(elcdf.loc[:,'mjd'])
        maxDate=np.max(elcdf.loc[:,'mjd'])

        halfPoint=np.average([minDate, maxDate])
        firstCut=np.average([minDate, halfPoint])
        secondCut=np.average([halfPoint, maxDate])
        minDate=np.min(elcdf.loc[:,'mjd'])
        maxDate=np.max(elcdf.loc[:,'mjd'])

        halfPoint=np.average([minDate, maxDate])
        firstCut=np.average([minDate, halfPoint])
        secondCut=np.average([halfPoint, maxDate])

        #early
        efilter=elcdf.loc[:,'mjd']<=firstCut
        #late
        lfilter=elcdf.loc[:,'mjd']>=secondCut
        #mid
        mfilter=(efilter | lfilter)==False
    
        edf=elcdf.loc[efilter]
        mdf=elcdf.loc[mfilter]
        ldf=elcdf.loc[lfilter]
        
        ledf = edf[edf['passband']==lep]
        hedf = edf[edf['passband']==hep]
        lmdf = mdf[mdf['passband']==lep]
        hmdf = mdf[mdf['passband']==hep]
        lldf = ldf[ldf['passband']==lep]
        hldf = ldf[ldf['passband']==hep]
    
    #using the datecutting method often leads to zero population sizes with non-ddf objects
    else:
        
        lowdf=elcdf[elcdf['passband']==lep]
        highdf=elcdf[elcdf['passband']==hep]
        lenLow=lowdf.shape[0]
        lenHigh=highdf.shape[0]
        
        minSizeLow = int(lenLow / 3)
        minSizeHigh = int(lenHigh / 3)
        
        lldf=lowdf.nlargest(minSizeLow, 'mjd')
        hldf=highdf.nlargest(minSizeHigh, 'mjd')
        ledf=lowdf.nsmallest(minSizeLow, 'mjd')
        hedf=highdf.nsmallest(minSizeHigh, 'mjd')
        lmdf=lowdf.nlargest(lenLow-minSizeLow, 'mjd').nsmallest(lenLow-2*minSizeLow, 'mjd')
        hmdf=highdf.nlargest(lenHigh-minSizeHigh, 'mjd').nsmallest(lenHigh-2*minSizeHigh, 'mjd')
    
    return ledf, hedf, lmdf, hmdf, lldf, hldf

aalcdf, balcdf, calcdf, dalcdf, ealcdf, falcdf=divideLcdf(alcdf, True)
print(aalcdf.shape)
print(balcdf.shape)
print(calcdf.shape)
print(dalcdf.shape)
print(ealcdf.shape)
print(falcdf.shape)
        


# In[ ]:


def getSubPopFeats(pbdf, outSig=3.0):
    
    average=np.average(pbdf.loc[:,'flux'])
    median=np.median(pbdf.loc[:,'flux'])
    stdev=np.std(pbdf.loc[:,'flux'])
    maxflux=np.max(pbdf.loc[:,'flux'])
    minflux=np.min(pbdf.loc[:,'flux'])
    stdflerr=np.std(pbdf.loc[:,'flux_err'])
    medflerr=np.median(pbdf.loc[:,'flux_err'])
    
    #We want a means to extract the rate of decay or rise from minima or maxima
    #This is grabbing within the population
    #We also will look between populations
    maxmjd=np.max(pbdf[pbdf['flux']==maxflux].loc[:,'mjd'])
    minmjd=np.max(pbdf[pbdf['flux']==minflux].loc[:,'mjd'])
    
    #at what date does the max occur?
    aftmaxdf=pbdf[pbdf['mjd']>maxmjd]
    
    #if there are data points after the max, what is the value and date of the lowest?
    if aftmaxdf.shape[0]>0:
        minaft=np.min(aftmaxdf.loc[:,'flux'])
        aftminmjd=np.min(aftmaxdf[aftmaxdf['flux']==minaft].loc[:,'mjd'])
        #(val at t0 - val at t1) / (t0 - t1) sb neg
        decaySlope=(maxflux-minaft)/(maxmjd-aftminmjd)
    
    else:
        decaySlope=0
        
    aftmindf=pbdf[pbdf['mjd']<minmjd]
    if aftmindf.shape[0]>0:
        maxaft=np.max(aftmindf.loc[:,'flux'])
        aftmaxmjd=np.max(aftmindf[aftmindf['flux']==maxaft].loc[:,'mjd'])
        #(val at t0 - val at t1) / (t0 - t1) sb pos
        riseSlope=(minflux - maxaft)/(aftmaxmjd-minmjd)
    
    else:
        riseSlope=0
        
    return average, stdev, median, medflerr, stdflerr, maxflux,             maxmjd, decaySlope, minflux, minmjd, riseSlope

a,b,c,d,e,f,g, h,i,j,k=getSubPopFeats(balcdf)
print(a)
print(b)
print(c)
print(d)
print(e)
print(f)
print(g)
print(h)
print(i)
print(j)
print(k)


# ### The processing isn't done, but we will do as much as possible array style later
# - TBD how much slowdown this change causes.  I am now returning 12 instead of 7
# - Want to get decay rates within subsection in case peak at last subsection

# In[ ]:


def processLc(objid, elcdf, ddf, lep=2, hep=5):
    
    lcdf=copy.deepcopy(elcdf)
    
    #feature borrowed from Grzegorz Sionkowski (../sionek)
    #dt[detected==1, mjd_diff:=max(mjd)-min(mjd), by=object_id]
    #detectMjds=elcdf[elcdf['detected']==1].loc[:,'mjd']
    #deltaDetect=np.max(detectMjds) - np.min(detectMjds)
    
    #divide the incoming light curve to 6 subpopulations
    ledf, hedf, lmdf, hmdf, lldf, hldf=divideLcdf(lcdf, ddf,lep=lep, hep=hep)
    #return average, stdev, median, medflerr, stdflerr, maxflux, \
    #        maxmjd, decayslope, minflux, minmjd, riseSlope
    
    leavg, lestd, lemed, lemfl, lesfl, lemax, lemxd, ledsl, lemin, lemnd, lersl=getSubPopFeats(ledf)
    heavg, hestd, hemed, hemfl, hesfl, hemax, hemxd, hedsl, hemin, hemnd, hersl=getSubPopFeats(hedf)
    lmavg, lmstd, lmmed, lmmfl, lmsfl, lmmax, lmmxd, lmdsl, lmmin, lmmnd, lmrsl=getSubPopFeats(lmdf)
    hmavg, hmstd, hmmed, hmmfl, hmsfl, hmmax, hmmxd, hmdsl, hmmin, hmmnd, hmrsl=getSubPopFeats(hmdf)
    llavg, llstd, llmed, llmfl, llsfl, llmax, llmxd, lldsl, llmin, llmnd, llrsl=getSubPopFeats(lldf)
    hlavg, hlstd, hlmed, hlmfl, hlsfl, hlmax, hlmxd, hldsl, hlmin, hlmnd, hlrsl=getSubPopFeats(hldf)
    
    
    feats= [objid, leavg, lestd, lemed, lemfl, lesfl, lemax, 
            lemxd, ledsl, lemin, lemnd, lersl,
            heavg, hestd, hemed, hemfl, hesfl, hemax, hemxd,
            hedsl, hemin, hemnd, hersl,
            lmavg, lmstd, lmmed, lmmfl, lmsfl, lmmax, lmmxd,
            lmdsl, lmmin, lmmnd, lmrsl,
            hmavg, hmstd, hmmed, hmmfl, hmsfl, hmmax, hmmxd, 
            hmdsl, hmmin, hmmnd, hmrsl,
            llavg, llstd, llmed, llmfl, llsfl, llmax, llmxd, 
            lldsl, llmin, llmnd, llrsl,
            hlavg, hlstd, hlmed, hlmfl, hlsfl, hlmax, hlmxd, 
            hldsl, hlmin, hlmnd, hlrsl]
    
    return feats

feats=processLc(objB, blcdf, bdf.loc[3000000,'ddf'])

print(feats)
    


# https://stackoverflow.com/questions/41888080/python-efficient-way-to-add-rows-to-dataframe#comment70958764_41888080
# 
# 
# 
# for row in iterable_object:
#     csv_writer.writerow(row)
# 
# return df

# In[ ]:


from io import StringIO
from csv import writer 
import time

def writeAChunk(firstRecord, lastRecord, info, statusFreq=500):
    output = StringIO()
    csv_writer = writer(output)

    fdf=pd.DataFrame(columns=['object_id', 'leavg', 'lestd', 'lemed', 'lemfl', 'lesfl', 'lemax', 
                'lemxd', 'ledsl', 'lemin', 'lemnd', 'lersl',
                'heavg', 'hestd', 'hemed', 'hemfl', 'hesfl', 'hemax', 'hemxd',
                'hedsl', 'hemin', 'hemnd', 'hersl',
                'lmavg', 'lmstd', 'lmmed', 'lmmfl', 'lmsfl', 'lmmax', 'lmmxd',
                'lmdsl', 'lmmin', 'lmmnd', 'lmrsl',
                'hmavg', 'hmstd', 'hmmed', 'hmmfl', 'hmsfl', 'hmmax', 'hmmxd', 
                'hmdsl', 'hmmin', 'hmmnd', 'hmrsl',
                'llavg', 'llstd', 'llmed', 'llmfl', 'llsfl', 'llmax', 'llmxd', 
                'lldsl', 'llmin', 'llmnd', 'llrsl',
                'hlavg', 'hlstd', 'hlmed', 'hlmfl', 'hlsfl', 'hlmax', 'hlmxd', 
                'hldsl', 'hlmin', 'hlmnd', 'hlrsl'])

    theColumns=fdf.columns
    
    csv_writer.writerow(theColumns)
    started=time.time()
    for rindex in range(firstRecord, lastRecord):
        #if you want to monitor progress
        #ddf 18 sec per 100 on my macAir
        #non ddf 25 sec per 100 on my macAir
        if rindex%statusFreq==(statusFreq-1):
            print(rindex)
            print("Processing took {:6.4f} secs, records = {}".format((time.time() - started), statusFreq))
            started=time.time()
            #fdf=pd.merge(fdf, tdf, on='key')
        objid = bdf.loc[rindex,'object_id']
        ddf=bdf.loc[rindex,'ddf']==1
        #ig=bdf.loc[rindex,'hostgal_specz']==0
        lcdf=getLCDF(objid, info)
        feats=processLc(objid, lcdf, ddf)
        #fdf.loc[rindex,:]=feats
        csv_writer.writerow(feats)

    output.seek(0) # we need to get back to the start of the BytesIO
    chdf = pd.read_csv(output)
    chdf.columns=theColumns
    
    return chdf

theColumns=['object_id', 'leavg', 'lestd', 'lemed', 'lemfl', 'lesfl', 'lemax', 
                'lemxd', 'ledsl', 'lemin', 'lemnd', 'lersl',
                'heavg', 'hestd', 'hemed', 'hemfl', 'hesfl', 'hemax', 'hemxd',
                'hedsl', 'hemin', 'hemnd', 'hersl',
                'lmavg', 'lmstd', 'lmmed', 'lmmfl', 'lmsfl', 'lmmax', 'lmmxd',
                'lmdsl', 'lmmin', 'lmmnd', 'lmrsl',
                'hmavg', 'hmstd', 'hmmed', 'hmmfl', 'hmsfl', 'hmmax', 'hmmxd', 
                'hmdsl', 'hmmin', 'hmmnd', 'hmrsl',
                'llavg', 'llstd', 'llmed', 'llmfl', 'llsfl', 'llmax', 'llmxd', 
                'lldsl', 'llmin', 'llmnd', 'llrsl',
                'hlavg', 'hlstd', 'hlmed', 'hlmfl', 'hlsfl', 'hlmax', 'hlmxd', 
                'hldsl', 'hlmin', 'hlmnd', 'hlrsl']

fdf=pd.DataFrame(columns=theColumns)
chunksize=700
firstLoop=0
lastLoop=100
loops=lastLoop-firstLoop
veryFirstRow=firstLoop*chunksize
veryLastRow=lastLoop*chunksize-1
for i in range(firstLoop, lastLoop):
    startRow=i*chunksize
    stopRow=(i+1)*chunksize
    chdf=writeAChunk(startRow, stopRow, info, statusFreq=int(chunksize/2))
    fdf= pd.concat([fdf, chdf])
    print(fdf.shape)


# **Merge features with header information**

# In[ ]:


print(bdf.shape)
print(fdf.shape)

#.merge complained without doing this
fdf.index=range(veryFirstRow,veryLastRow+1)
fdf.loc[:,'object_id']=fdf.loc[:,'object_id'].astype(str)
bdf.loc[:,'object_id']=bdf.loc[:,'object_id'].astype(str)

mdf=fdf.merge(bdf, on='object_id', how='left')
print(mdf.shape)
mdf.head()


# **Psuedo box-plot**
# - checking to see if populations overlap one another
# - transients can slope up, down, or peak in middle
# - but at least one of early, middle, late will look different from others
# - especially if repeated in both passbands

# In[ ]:


def testForOutlier(bdf, energy='high', sigmas=1.0):
    
    if energy=='high':
        valCols=['heavg', 'hmavg', 'hlavg']
        sigCols=['hestd', 'hmstd', 'hlstd']
    else:
        valCols=['leavg', 'lmavg', 'llavg']
        sigCols=['lestd', 'lmstd', 'llstd']
    
    fdf=copy.deepcopy(bdf)
    

    
    fdf.loc[:,energy + 'Energy_transitory_' + str(round(sigmas,1)) + '_TF']=False
    for i in range(len(valCols)):
        fdf.loc[:,'min' + str(valCols[i])] = fdf.loc[:,valCols[i]] - sigmas*fdf.loc[:,sigCols[i]]
        fdf.loc[:,'max' + str(valCols[i])] = fdf.loc[:,valCols[i]] + sigmas*fdf.loc[:,sigCols[i]]
    
    for i in range(len(valCols)):
        #fdf.loc[:,'earlySet']=range(fdf.loc[:,'minX100' + str(valCols[0])],fdf.loc[:, 'maxX100' + str(valCols[0])])
        #earlyMaxLessThanMedMin
        for j in range(len(valCols)):
            if j!=i:
                maxFailsOverlap=fdf.loc[:,'max' + str(valCols[i])]<fdf.loc[:,'min' + str(valCols[j])]
                minFailsOverlap=fdf.loc[:,'min' + str(valCols[i])]>fdf.loc[:,'max' + str(valCols[j])]
                fdf.loc[:,energy + 'Energy_transitory_' + str(round(sigmas,1)) + '_TF']=                 (fdf.loc[:,energy + 'Energy_transitory_' + str(round(sigmas,1)) + '_TF'] | minFailsOverlap | maxFailsOverlap)
    
    for i in range(len(valCols)):
        fdf=fdf.drop('min' + str(valCols[i]), axis=1)
        fdf=fdf.drop('max' + str(valCols[i]), axis=1)
        
    return fdf


energy='high'
sigmas=1.0
fdf=testForOutlier(mdf)
fdf.shape
print(fdf.loc[:,energy + 'Energy_transitory_' + str(round(sigmas,1)) + '_TF'].sum())

sigmas=1.5
fdf=testForOutlier(fdf, energy=energy, sigmas=sigmas)
print(fdf.loc[:,energy + 'Energy_transitory_' + str(round(sigmas,1)) + '_TF'].sum())

energy='low'
sigmas=1.0
fdf=testForOutlier(fdf, energy=energy, sigmas=sigmas)
print(fdf.loc[:,energy + 'Energy_transitory_' + str(round(sigmas,1)) + '_TF'].sum())

sigmas=1.5
fdf=testForOutlier(fdf, energy=energy, sigmas=sigmas)
print(fdf.loc[:,energy + 'Energy_transitory_' + str(round(sigmas,1)) + '_TF'].sum())


# **Consolidate outlier information**

# In[ ]:




fdf.loc[:,'outlierString']=fdf.loc[:,'highEnergy_transitory_1.5_TF'].astype(str) +                              fdf.loc[:,'highEnergy_transitory_1.0_TF'].astype(str) +                              fdf.loc[:,'lowEnergy_transitory_1.5_TF'].astype(str) +                              fdf.loc[:,'lowEnergy_transitory_1.0_TF'].astype(str)


def getOutlierScore(row):
    tdict={'TrueTrueTrueTrue':8, 'FalseTrueTrueTrue':7, 'TrueTrueFalseTrue':7,
       'FalseTrueFalseTrue':6, 'FalseFalseTrueTrue':3, 'TrueTrueFalseFalse':3,
       'FalseTrueFalseFalse':3, 'FalseFalseFalseTrue':3, 'FalseFalseFalseFalse':0}
    
    #tdict could be a parameter
    #both very significant - 8
    #one fairly, one very - 7
    #both fairly, 6

    #the reason for the dropoff from both fairly to one very is that a deviation in one band could be measurement error
    #the reason it doesn't drop lower is that the lack of deviation in one band could ALSO be measurement error
    #the reason one fairly = one very is that measurement errors are likely to have high magnitude
    #one very 3
    #one fairly 3
    #none 0

    return tdict[row['outlierString']]

fdf['outlierScore']=fdf.apply(getOutlierScore, axis=1)
    
fdf=fdf.drop('outlierString', axis=1)

#fdf.to_csv('fastestFeatureTableWithTransitoryFlags.csv')
print(fdf.shape)
print(fdf.columns)
print(np.average(fdf.loc[:,'outlierScore']))
print(np.min(fdf.loc[:,'outlierScore']))
print(np.max(fdf.loc[:,'outlierScore']))
print(np.median(fdf.loc[:,'outlierScore']))


# ## Rate of decay
# - If outlierScore is zero we only care about the intraPop slopes and its amplitude related
# - If it is non-zero we want to see 

# In[ ]:


fdf['hipd']=0
fdf['hipr']=0
fdf['htpd']=0
fdf['htpr']=0

fdf['lipd']=0
fdf['lipr']=0
fdf['ltpd']=0
fdf['ltpr']=0

outlierFilter=(fdf['outlierScore']>0)
print(outlierFilter.sum())

hipdFilter = (fdf['hmmax']>fdf['hemax']) & (fdf['hmmax']>fdf['hlmax']) & outlierFilter
htpdFilter = (fdf['hemax']>fdf['hmmax']) & (fdf['hemax']>fdf['hlmax']) & outlierFilter
lipdFilter = (fdf['lmmax']>fdf['lemax']) & (fdf['lmmax']>fdf['llmax']) & outlierFilter
ltpdFilter = (fdf['lemax']>fdf['lmmax']) & (fdf['lemax']>fdf['llmax']) & outlierFilter

print(hipdFilter.sum())
print(htpdFilter.sum())
print(lipdFilter.sum())
print(ltpdFilter.sum())


# ## Use the filters to set decay values without ifs and loops

# In[ ]:


#peak to peak
#these are light curves where the peak was in the middle
fdf.loc[hipdFilter,'hipd']=(fdf.loc[hipdFilter,'hmmax']-fdf.loc[hipdFilter,'hlmax']) /      (fdf.loc[hipdFilter,'hmmxd']-fdf.loc[hipdFilter,'hlmxd'])
fdf.loc[lipdFilter,'lipd']=(fdf.loc[lipdFilter,'lmmax']-fdf.loc[lipdFilter,'llmax']) /      (fdf.loc[lipdFilter,'lmmxd']-fdf.loc[lipdFilter,'llmxd'])

#these are light curves where the peak was in the beginning
fdf.loc[htpdFilter,'hipd']=(fdf.loc[htpdFilter,'hemax']-fdf.loc[htpdFilter,'hmmax']) /      (fdf.loc[htpdFilter,'hemxd']-fdf.loc[htpdFilter,'hmmxd'])
fdf.loc[ltpdFilter,'lipd']=(fdf.loc[ltpdFilter,'lemax']-fdf.loc[ltpdFilter,'lmmax']) /      (fdf.loc[ltpdFilter,'lemxd']-fdf.loc[ltpdFilter,'lmmxd'])
fdf.loc[htpdFilter,'htpd']=(fdf.loc[htpdFilter,'hmmax']-fdf.loc[htpdFilter,'hlmax']) /      (fdf.loc[htpdFilter,'hmmxd']-fdf.loc[htpdFilter,'hlmxd'])
fdf.loc[ltpdFilter,'ltpd']=(fdf.loc[ltpdFilter,'lmmax']-fdf.loc[ltpdFilter,'llmax']) /      (fdf.loc[ltpdFilter,'lmmxd']-fdf.loc[ltpdFilter,'llmxd'])

#print(fdf.loc[lipdFilter,'lipd'])
fdf[outlierFilter].head()


# ## Repeat decay stats for rise
# - Not sure if this will have significance but not ruling it out
# - Many of the rises will have negative slopes - which is counterintuitive at first but is because most outliers are flux spikes so even min to min is a decay
# - If an outlier is actually a trough then the 'decay' could have a positive slope

# In[ ]:


hiprFilter = (fdf['hmmin']<fdf['hemin']) & (fdf['hmmin']<fdf['hlmin']) & outlierFilter
htprFilter = (fdf['hemin']<fdf['hmmin']) & (fdf['hemin']<fdf['hlmin']) & outlierFilter
liprFilter = (fdf['lmmin']<fdf['lemin']) & (fdf['lmmin']<fdf['llmin']) & outlierFilter
ltprFilter = (fdf['lemin']<fdf['lmmin']) & (fdf['lemin']<fdf['llmin']) & outlierFilter

#these are light curves where the peak was in the middle
fdf.loc[hipdFilter,'hipr']=(fdf.loc[hipdFilter,'hmmin']-fdf.loc[hipdFilter,'hlmin']) /      (fdf.loc[hipdFilter,'hmmnd']-fdf.loc[hipdFilter,'hlmnd'])
fdf.loc[lipdFilter,'lipr']=(fdf.loc[lipdFilter,'lmmin']-fdf.loc[lipdFilter,'llmin']) /      (fdf.loc[lipdFilter,'lmmnd']-fdf.loc[lipdFilter,'llmnd'])

#these are light curves where the peak was in the beginning
fdf.loc[htpdFilter,'hipr']=(fdf.loc[htpdFilter,'hemin']-fdf.loc[htpdFilter,'hmmin']) /      (fdf.loc[htpdFilter,'hemnd']-fdf.loc[htpdFilter,'hmmnd'])
fdf.loc[ltpdFilter,'lipr']=(fdf.loc[ltpdFilter,'lemin']-fdf.loc[ltpdFilter,'lmmin']) /      (fdf.loc[ltpdFilter,'lemnd']-fdf.loc[ltpdFilter,'lmmnd'])
fdf.loc[htpdFilter,'htpr']=(fdf.loc[htpdFilter,'hmmin']-fdf.loc[htpdFilter,'hlmin']) /      (fdf.loc[htpdFilter,'hmmnd']-fdf.loc[htpdFilter,'hlmnd'])
fdf.loc[ltpdFilter,'ltpr']=(fdf.loc[ltpdFilter,'lmmin']-fdf.loc[ltpdFilter,'llmin']) /      (fdf.loc[ltpdFilter,'lmmnd']-fdf.loc[ltpdFilter,'llmnd'])

fdf[outlierFilter].head()


# save output

# In[ ]:


fdf.to_csv('testFeaturesFrom' + str(veryFirstRow) + 'TO' + str(veryLastRow) + '.csv')
print('saved file: testFeaturesFrom_' + str(veryFirstRow) + '_to_' + str(veryLastRow) + '.csv')


# 
