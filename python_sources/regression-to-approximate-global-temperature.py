import csv
import warnings
import numpy
import pylab
import random
import math

class PolynomialRegression(object):

	def __init__(self, trainingSet, testSet):
		self.x = numpy.array(trainingSet.x)
		self.y = numpy.array(trainingSet.y)

		self.testX = numpy.array(testSet.x)
		self.testY = numpy.array(testSet.y)

	def getPolynomial(self, degree, x1, y1):
		z = numpy.polyfit(x1, y1, degree)
		return z

	def validation(self, z, x1, y1):
		fitFunc = numpy.poly1d(z)
		y_fit = [];
		for x_val in x1:
			y_fit.append(fitFunc(x_val))

		err = 0
		for i in range(len(y_fit)):
			err = err + pow((y1[i]-y_fit[i]), 2)
		err = math.sqrt(err)/len(y_fit)
		return err

	def performRegression(self):
		testErrors = []
		trainingErrors = []
		hlist = []

		for i in range(20):
			trainingPolynomial = self.getPolynomial(i, self.x, self.y)
			hlist.append(trainingPolynomial)
			trainingError = self.validation(trainingPolynomial, self.x, self.y)
			trainingErrors.append(trainingError)
			testError = self.validation(trainingPolynomial, self.testX, self.testY)
			testErrors.append(testError)

		minErrorDegree = testErrors.index(min(testErrors))
		# print trainingErrors, testErrors
		# print "Degree = ", minErrorDegree, "; minimum error = ", min(testErrors)
		finalfunc = hlist[minErrorDegree]

		# xx = numpy.linspace(0,54,55)
		# pylab.plot(xx, trainingErrors, '-g', label="Training")
		# pylab.plot(xx, testErrors, '-r', label="Test")
		# pylab.legend()
		# pylab.show()
		
		return finalfunc


""" Stores the data of month-wise average temperatures
"""
class TemperatureRegression:

	def __init__(self):
		self.jan = {}
		self.feb = {}
		self.march = {}
		self.april = {}
		self.may = {}
		self.june = {}
		self.july = {}
		self.aug = {}
		self.sept = {}
		self.octo = {}
		self.nov = {}
		self.dec = {}

	# Adds <year>:<average temperature> data to a particular month array
	def addDataToArray(self, temperature, month, year):
		switcher = {
			1: self.jan,
			2: self.feb,
			3: self.march,
			4: self.april,
			5: self.may,
			6: self.june,
			7: self.july,
			8: self.aug,
			9: self.sept,
			10:self.octo,
			11:self.nov,
			12:self.dec
		}
		monthDict = switcher[int(month)]
		monthDict[int(year)] = float(temperature)

	# Returns array corresponding to the month number
	def getMonthArray(self, month):
		switcher = {
			1: self.jan,
			2: self.feb,
			3: self.march,
			4: self.april,
			5: self.may,
			6: self.june,
			7: self.july,
			8: self.aug,
			9: self.sept,
			10:self.octo,
			11:self.nov,
			12:self.dec
		}
		return switcher[int(month)]
		
	def getMonthName(self, month):
		switcher = {
			1: "Jan",
			2: "Feb",
			3: "March",
			4: "April",
			5: "May",
			6: "June",
			7: "July",
			8: "Aug",
			9: "Sept",
			10:"Oct",
			11:"Nov",
			12:"Dec"
		}
		return switcher[int(month)]


class Dataset:
	x = []
	y = []


TIME = 0
GLOBAL_AVG = 1
EU_AVG = 2
YEAR = 3
MONTH = 4

""" Read the training set and test set in 2 CSV files. Info read:
1) Month, year
2) Global average temperature
3) European average temperature

Use this data to perform polynomial regression and pick the one with minimum cross validation error
"""

warnings.simplefilter('ignore', numpy.RankWarning)

f1 = open("../input/Data_for_month_10_2016_plot_3.csv", 'rt')
globalData = TemperatureRegression()
europeData = TemperatureRegression()
globalDataTest = TemperatureRegression()
europeDataTest = TemperatureRegression()
globalDataFull = TemperatureRegression()
europeDataFull = TemperatureRegression()

"""Reads the training set from a CSV"""
firstline = True
try:
	reader = csv.reader(f1)
	for row in reader :
		if firstline:
			firstline = False
			continue
		globalAvg = row[GLOBAL_AVG]
		euAvg = row[EU_AVG]
		year = int(int(row[TIME])/100)
		month = int(row[TIME])%100

		globalData.addDataToArray(globalAvg, month, year)
		europeData.addDataToArray(euAvg, month, year)
		globalDataFull.addDataToArray(globalAvg, month, year)
		europeDataFull.addDataToArray(euAvg, month, year)
finally:
	f1.close()
"""Iterates over each month and generates 10 random number between the range of 1979, 2016 (data is available in this range of years for each month)
10 random years are selected, removed from training set and added to the test set"""
for i in range(1,13):
	mdg = globalData.getMonthArray(i)
	mdeu = europeData.getMonthArray(i)
	testIndices = random.sample(range(1979,2016), 10)
	for index in testIndices:
		gvalue = mdg.pop(index)
		euvalue = mdeu.pop(index)
		globalDataTest.addDataToArray(gvalue, i, index)
		europeDataTest.addDataToArray(euvalue, i, index)


"""Performing polynomial regression for each month"""
for i in range(1,13):
	monthDictGlobal = globalData.getMonthArray(i)
	monthDictTestGlobal = globalDataTest.getMonthArray(i)
	monthDictEu = europeData.getMonthArray(i)
	monthDictTestEu = europeDataTest.getMonthArray(i)
	
	trainingSetGlobal = Dataset()
	testSetGlobal = Dataset()
	trainingSetEu = Dataset()
	testSetEu = Dataset()
	fullSetGlobal = Dataset()
	fullSetEu = Dataset()

	globalX = []
	globalY = []
	globalTestX = []
	globalTestY = []
	euX = []
	euY = []
	euTestX = []
	euTestY = []

	for key, value in monthDictGlobal.items():
		globalX.append(key)
		globalY.append(value)

	for key, value in monthDictEu.items():
		euX.append(key)
		euY.append(value)

	for key, value in monthDictTestGlobal.items():
		globalTestX.append(key)
		globalTestY.append(value)

	for key, value in monthDictTestEu.items():
		euTestX.append(key)
		euTestY.append(value)

	trainingSetGlobal.x = globalX
	trainingSetGlobal.y = globalY
	trainingSetEu.x = euX
	trainingSetEu.y = euY
	testSetGlobal.x = globalTestX
	testSetGlobal.y = globalTestY
	testSetEu.x = euTestX
	testSetEu.y = euTestY

	fullSetGlobal.x = globalX + globalTestX
	fullSetGlobal.y = globalY + globalTestY
	fullSetEu.x = euX + euTestX
	fullSetEu.y = euY + euTestY

	prglobal = PolynomialRegression(trainingSetGlobal, testSetGlobal)
	funcGlobal = prglobal.performRegression()
	finalfuncGlobal = numpy.poly1d(funcGlobal)
	prEu = PolynomialRegression(trainingSetEu, testSetEu)
	funcEu = prEu.performRegression()
	finalfuncEu = numpy.poly1d(funcEu)
	xx = numpy.linspace(1979,2016,4000)
	finalfuncZ = numpy.poly1d(finalfuncGlobal)
	pylab.plot(xx, finalfuncGlobal(xx), '-g', label="Approx fun (Global)")
	# pylab.plot(xx, finalfuncEu(xx), '-r', label="Europe")
	pylab.plot(fullSetGlobal.x, fullSetGlobal.y, 'o g', label="Global")
	pylab.plot(fullSetEu.x, fullSetEu.y, 'o r', label="Europe")
	pylab.title(globalData.getMonthName(i))
	pylab.legend()
	figname = str(i) + ".png"
	pylab.savefig(figname)
	pylab.clf()
	pylab.close()

	# xx = numpy.linspace(1979,2016,4000)
	# finalfuncZ = numpy.poly1d(finalfuncEu)
	# pylab.plot(xx, finalfuncEu(xx))
	# pylab.show()
