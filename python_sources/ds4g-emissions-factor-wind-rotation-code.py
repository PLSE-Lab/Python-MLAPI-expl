#
#  For each power plant (in a list that's a subset derived from gppd_120_pr)
#  * rotate all suitable S5P NO2 pictures about the power plant location, with an angle
#    based on the wind speed so that the wind direction is on the negative X axis for all.
#  * sum the rotated pictures and extract a section of the summed picture around the power plant
#  * calculate the average NO2 concentration for this piece wind-aligned photo
#  * fit an exponential to the decaying part of this piece to derive an NO2 lifetime
#
#
#
import numpy as np
import pandas as pd
import os
from skimage.transform import rotate
import matplotlib.pyplot as plt
import time
from sklearn.impute import SimpleImputer
import seaborn as sns
import rasterio as rio
from osgeo import gdal
import rasterio.plot
import rasterio.warp
from scipy.optimize import curve_fit


zerosImputer = SimpleImputer(missing_values=0.0, strategy='mean') # replaces zeros
NaNImputer = SimpleImputer(missing_values=np.nan, strategy='mean') # replaces NaN's


start = time.time()

#
#  construct a name for the closest GFS file (which would be at hour 18)
#
def GFSfileName(imageFileName):
   picYear = int(imageFileName[8:12])
   picMonth = int(imageFileName[12:14])
   picMonthString = "{:02d}".format(picMonth)
   picDay = int(imageFileName[14:16])
   picDayString = "{:02d}".format(picDay)
   weatherFileName = 'gfs_' + str(picYear)+picMonthString+picDayString+"18.tif"
   return(weatherFileName)

#
#   Get wind information from the GFS data closest in time (18:00) to the NO2 pictures (~17:00)
#   Uses spatial indexing to the information from the pixel that contains the plant location
#
def getWindComponents(imageFileName, plantPointLL):
    GFSdataset = rio.open(GFSDirectory + GFSfileName(imageFileName))
    UWindArray = GFSdataset.read(4)
    VWindArray = GFSdataset.read(5)
    GFSrow, GFScol = GFSdataset.index(plantPointLL['coordinates'][0],plantPointLL['coordinates'][1])
    UWind = UWindArray[GFSrow, GFScol]
    VWind = VWindArray[GFSrow, GFScol]
    return(UWind, VWind)

#
#   Transform the input image from lat/long to meters, and also transform
#   the plant location the same way. Return the row,col of the plant location
#   in the new picture.
#    
def rotateOnePicture(imageFileName, imageFileDataset, plantPointLL):
    UWind, VWind = getWindComponents(imageFileName, plantPointLL) # get wind direction
    windSpeed = np.sqrt(np.square(UWind) + np.square(VWind))
    
    plantRow, plantCol = imageFileDataset.index(plantLongitude,plantLatitude)
    
    imageBand = imageFileDataset.read(2)  # tropospheric NO2
    
    NaNFraction = np.sum(np.isnan(imageBand ))/(imageBand.shape[0]*imageBand.shape[1])   
    #
    # Warp the dataset to change it from lat,long to meters
    #
    fullPath = NO2Directory + imageFileName

    gdal.Warp('new.tif', fullPath, dstSRS='EPSG:32619', srcSRS='epsg:4326')
    
    newRasterDataset = rio.open('new.tif') # open warped dataset
    
    newRaster = newRasterDataset.read(2) # read warped raster back in
    
    tropoNO2Pic = NaNImputer.fit_transform(newRaster)  # use imputer to fill in NaN's
    
    plantPointMeters = rio.warp.transform_geom(src_crs='epsg:4326', dst_crs='EPSG:32619', geom=plantPointLL)
    
    row, col = newRasterDataset.index(plantPointMeters['coordinates'][0],plantPointMeters['coordinates'][1])
#    print(row,col)
    newRasterDataset.close()
    
    rotationAngleDegrees = 180. -(180./np.pi)*np.arctan2(VWind, UWind)
#    print("UWind", UWind,"VWind", VWind, "rotationAngleDegrees", rotationAngleDegrees)
    rotatedPic = rotate(tropoNO2Pic, rotationAngleDegrees, center=(col,row))
    return(rotatedPic, row, col, windSpeed)

#
#   An exponential decay function with spatial decay constant lengthScale
#
def fitFunction(x, a, lengthScale):
    g = a * np.exp(-x/lengthScale)
    return(g)


#
#   Extract a piece of the picture near the plant which is at plantRow,plantCol
#   in the transformed-to-meters averaged picture sumPic.  The piece should contain the NO2 peak
#   plus enough of the exponential drop-off to fit the NO2 lifetime.
#   Fit that exponential starting a ways off the peak NO2.
#   Returns the fit parameters a and lengthScale, and the total NO2 in this reduced area.
#
def fitSumPic(sumPic, plantRow, plantCol):
    maxRow = sumPic.shape[0]
    maxCol = sumPic.shape[1]
    
    xStartOffset = 40
    fitAreaMinRow = max(0, plantRow-20)
    fitAreaMaxRow = min(maxRow, plantRow+20)
    fitAreaMinCol = max(0, plantCol-250)
    fitAreaMaxCol = min(maxCol, plantCol+xStartOffset)
    
    fitArea = sumPic[fitAreaMinRow:fitAreaMaxRow, fitAreaMinCol:fitAreaMaxCol]
    NO2sumOfFitArea = np.sum(fitArea)
    pixelWidthOfFitArea = fitAreaMaxCol - fitAreaMinCol
    pixelsInFitArea = fitArea.shape[0] * fitArea.shape[1]
    
    colSums = np.array([np.sum(fitArea[:,i]) for i in range(fitArea.shape[1])])
    
    xData = np.array(range(len(colSums))) - 50.
    yData = np.flip(colSums).astype(np.float) # only needed if colSums is INT
#    plt.figure()
#    plt.plot(xData, yData)

    params, covMatrix = curve_fit(fitFunction, xData[50:], yData[50:])
#    plt.plot(xData, fitFunction(xData, *params), 'r-',
#         label='fit: a=%5.3f, lengthScale=%5.3f' % tuple(params))
#    plt.show()
    return(params[0], params[1], NO2sumOfFitArea, pixelWidthOfFitArea, pixelsInFitArea)

#
#     MAIN PROGRAM
#
#
NO2Directory = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/
GFSDirectory = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/

#
# Get total Puerto Rico electrical consumption for period of this study
# Uses public kaggle dataset "DS4G Puerto Rico Electricity Consumption by month"
#
elec = pd.read_csv('../input/elec-consumption-same-as-public-dataset/elec_consumption_same_as_public_dataset.csv')

elecForStudyTimePeriod = elec.query("yearmonth>=201807 and yearmonth<201907")
#print("EIA Puerto Rico Electric Consumption by Month for Study Period")
#print(elecForStudyTimePeriod[['yearmonth','all_sectors']])
elecConsumption = elecForStudyTimePeriod['all_sectors'].sum()
print("Total PR Electric Consumption for study Time Period", elecConsumption,"GWh")

nameList = [] # power-plant names
NPixList = [] # number of pictures used for each power plant
aList = [] # "a" parameter of the exponential fit
tauList = [] # tau, the NO2 lifetime calculated from the fit of an exponential
tropoNO2ktList = []
NO2AvgConcentrationList = []
windSpeedList = []
ktPerYearAssumingTauList = []
NO2EmissionsFactorList = []
#
#  MAIN LOOP OVER ALL POWER PLANTS
#
#gppd = pd.read_csv('junk.csv')
gppd = pd.read_csv('gppd_subset.csv')

for i in range(len(gppd)):
    plantName = gppd.loc[i,'name']
    plantLongitude = gppd.loc[i,'longitude'] # Palo Seco
    plantLatitude =  gppd.loc[i,'latitude'] # Palo Seco
        
    plantPointLL = {'type': 'Point',
                    'coordinates': [plantLongitude, plantLatitude]}
#
#   Loop over image files.  Skip the ones with a lot of NaN's.
#    
    first = True
    NPix = 1
    windSpeedSum = 0 # want to average wind speeds at the end
    for imageFileName in os.listdir(NO2Directory):
        imageFileDataset = rio.open(NO2Directory + imageFileName) # epsg 4326
        imageBand = imageFileDataset.read(2)  # tropospheric NO2
        NaNFraction = np.sum(np.isnan(imageBand ))/(imageBand.shape[0]*imageBand.shape[1])
    #    print('NaNFraction',NaNFraction)
    
        if NaNFraction < 0.06:
           rotPic, plantRow, plantCol, windSpeed = rotateOnePicture(imageFileName, imageFileDataset, plantPointLL)
           windSpeedSum += windSpeed
           rotZeroFraction = np.sum(rotPic==0.0)/(rotPic.shape[0]*rotPic.shape[1])
           rotNaNFraction = np.sum(np.isnan(rotPic))/(rotPic.shape[0]*rotPic.shape[1])       
    #       print('rotZeroFraction', rotZeroFraction, 'NaNFraction', rotNaNFraction)
    #       plt.figure()
    #       sns.heatmap(rotPic, cmap='inferno')
    #       plt.show()
           if first:
               sumPic = rotPic.copy() # initialize summed picture
               first = False
           else:
               sumPic += rotPic
           NPix += 1
    
    print('NPix',NPix)
    windSpeedAvg = windSpeedSum / NPix
    a, lengthScale, NO2sumOfFitArea, pixelWidthOfFitArea, pixelsInFitArea = fitSumPic(sumPic, plantRow, plantCol) # fit exponential
#    print("a, lengthScale, NO2sumOfFitArea, pixelsInFitArea", a, lengthScale, NO2sumOfFitArea, pixelsInFitArea)
    
    
    #
    #  Use last saved picture (coords in meters) to recall the area and width of the picture
    #
    lastPicSavedInMeters = rio.open('new.tif')
    bounds = lastPicSavedInMeters.bounds
    width = bounds[2]-bounds[0]
    height= bounds[3] - bounds[1]
    finalPicSquareMeters = height * width
#    print('finalPicSquareMeters',finalPicSquareMeters)
    finalPicShape = lastPicSavedInMeters.read(2).shape
    pixelsInFinalPic = finalPicShape[0] * finalPicShape[1]
    pixelWidthInMeters = width/finalPicShape[1]
    pixelHeightInMeters = height/finalPicShape[0] 
    lastPicSavedInMeters.close()
    
    NO2AvgConcentration = NO2sumOfFitArea / NPix
    tropoNO2totalMoles =  NO2AvgConcentration * finalPicSquareMeters * pixelsInFitArea/pixelsInFinalPic
    NO2MolecularWeight = 2*15.999 + 14.007  # 2 oxygen plus 1 nitrogen
    tropoNO2totalGrams = NO2MolecularWeight * tropoNO2totalMoles
    gramsPerKilotonne = 1000 * 1000 * 1000   # 1000g/kg, 1000kg/tonne, 1000tonne/kt
    tropoNO2kt = tropoNO2totalGrams / gramsPerKilotonne 


    l = lengthScale * pixelWidthInMeters # convert lengthScale from fit to meters
    print('lengthScale,pixelWidthInMeters', lengthScale,pixelWidthInMeters)
    tauSeconds = l/windSpeed
    tau = l/windSpeed/3600 # lifetime of NO2 converted to hours

    assumedTauInHours = 6 # 6 hours, from Fioletov 2016
    ktPerYearAssumingTau = (tropoNO2kt/assumedTauInHours) * 365*24
#    NO2EmissionsFactor = ktPerYearAssumingTau / elecConsumption # need elec by plant to do this right
    
    nameList.append(plantName)
    aList.append(a) # "a" parameter of the exponential fit
    tauList.append(tau) # tau, the NO2 lifetime calculated from the fit of an exponential
    tropoNO2ktList.append(tropoNO2kt)
    NO2AvgConcentrationList.append(NO2AvgConcentration)
    windSpeedList.append(windSpeedAvg)
    ktPerYearAssumingTauList.append(ktPerYearAssumingTau)
#    NO2EmissionsFactorList.append(NO2EmissionsFactor)

#    plt.figure()
#    sns.heatmap(sumPic, cmap='inferno')
#    plt.show()
    
    np.save('sumPic'+plantName+'.npy',sumPic)
    plt.imsave('sumPic'+plantName+'.tiff',sumPic,cmap='gray')
    
    print(plantName,"is finished.")

#
# Make a dataframe from the lists compiled above
#
output = pd.DataFrame()
output['plantName'] = nameList
output['a'] = aList
output['tau'] = tauList
output['windSpeed'] = windSpeedList
output['tropoNO2kt'] = tropoNO2ktList
output['NO2AvgConcentration'] = NO2AvgConcentrationList
output['windSpeed'] = windSpeedList
output['ktPerYearAssumingTau'] = ktPerYearAssumingTauList
#output['NO2EmissionsFactor'] = NO2EmissionsFactorList
output.to_csv('model_output.csv')

end = time.time()
print("time taken:", end-start)
