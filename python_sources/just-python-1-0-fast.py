import codecs
import csv
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from tqdm import tqdm
import time



if (__name__ == '__main__'):

	startTime = time.time()

	lstOrders = []
	setShopIDs = set()

	INPUT_CSV = "../input/order-brushing-shopee-code-league/order_brush_order.csv"
	with codecs.open(INPUT_CSV, 'r', encoding = 'utf-8', errors = 'ignore') as input_file:

		lines = input_file.readlines()
		csvReader = csv.reader(lines, delimiter = ",", quotechar = '"')

		# Skip the header row
		next(csvReader)

		for orderTuple in csvReader:

			orderID, shopID, userID, orderTime = orderTuple
			# print("Order ID: {:30s} Shop ID: {:20s} User ID: {:20s} Order Time: {:20s}".format( orderID, shopID, userID, orderTime ))

			lstOrders.append( [shopID.strip(), userID.strip(), orderID.strip(), orderTime.strip()] )
			setShopIDs.add(shopID.strip())

	print("\n# of Orders: {:,d}".format( len(lstOrders) )) 	# 222,750
	print("# of Shops: {:,d}".format( len(setShopIDs) )) 	# 18,770



	shopID_2_allOrders = defaultdict(list)

	for (shopID, userID, orderID, orderTime) in lstOrders:

		# print("Order ID: {:30s} Shop ID: {:20s} User ID: {:20s} Order Time: {:20s}".format( orderID, shopID, userID, orderTime ))

		shopID_2_allOrders[shopID].append( [userID, orderID, orderTime] )

	print("\n# of Shops: {:,d}".format( len(shopID_2_allOrders.keys()) ))
	print("# of Orders: {:,d}".format( sum([len(x) for x in shopID_2_allOrders.values()]) ))

	print("\nThe Orders have been grouped by the Shops...")



	timeFmtStr = "%Y-%m-%d %H:%M:%S"
	shopID_2_dictChronoOrders = defaultdict()

	for shopID, allOrders in shopID_2_allOrders.items():

		if (len(allOrders) < 3):
			continue

		chronoOrders = []
		for (currUserID, currOrderID, currOrderTimeStr) in allOrders:

			currOrderDateTime = datetime.strptime(currOrderTimeStr, timeFmtStr)
			chronoOrders.append( [currUserID, currOrderID, currOrderDateTime] )

		chronoOrders.sort(key = lambda x: x[2])

		# ***** NOTE: Use Python 3.7+ *****
		# "built-in dict class gained the ability to remember insertion order (this new behavior became guaranteed in Python 3.7)"
		dictChronoOrders = defaultdict(list)
		for (currUserID, currOrderID, currOrderDateTime) in chronoOrders:
			dictChronoOrders[currOrderDateTime].append( [currUserID, currOrderID] )

		shopID_2_dictChronoOrders[shopID] = dictChronoOrders

	del shopID_2_allOrders
	print("\nThe Orders for each Shop have been ordered chronologically, and grouped by timestamps...")



	shopID_2_badOrders = defaultdict(list)
	shopID_2_badOrderIDs = defaultdict(set)

	print("")
	for shopID, dictChronoOrders in tqdm(shopID_2_dictChronoOrders.items(), desc = "Finding Bad Orders"):

		# Convert to a list (for slicing)
		listChronoOrders = [(k, v) for k, v in dictChronoOrders.items()]

		# The following list of UNIQUE timestamps is chronologically sorted
		lstUniqueTimestamps = [x[0] for x in listChronoOrders]
		finalTimestamp = lstUniqueTimestamps[-1]

		oneHour = timedelta(seconds = 3600)

		# Timestamp to track the sliding 'temporal window'
		currTemporalMarker = lstUniqueTimestamps[0] - oneHour

		for timeIdx, (t, setOrders) in enumerate(listChronoOrders):

			lstOrdersWithinNextHour = []
			latestOrderDateTime = t + oneHour

			for candTimeIdx, (candT, candSetOrders) in enumerate(listChronoOrders[timeIdx:]):

				if (candT > latestOrderDateTime):
					break

				if (candT <= latestOrderDateTime):
					lstOrdersWithinNextHour.extend(candSetOrders)

				if (candT == finalTimestamp or lstUniqueTimestamps[timeIdx + candTimeIdx + 1] > currTemporalMarker + oneHour):

					# Get unique User IDs
					setPeriodUserIDs = set([x[0] for x in lstOrdersWithinNextHour])

					# Calculate 'concentrate rate': Number of Orders / Number of Unique Users
					concentrateRate = len(lstOrdersWithinNextHour) / len(setPeriodUserIDs)

					if (concentrateRate >= 3.0):

						newBadOrders = [x for x in lstOrdersWithinNextHour if x[1] not in shopID_2_badOrderIDs[shopID]]
						shopID_2_badOrders[shopID].extend( newBadOrders )
						shopID_2_badOrderIDs[shopID].update( [x[1] for x in newBadOrders] )

			# Update the current 'last checked' timestamp
			currTemporalMarker = t



	shopID_2_baddestUsers = defaultdict(list)

	print("")
	for shopID, badOrders in tqdm(shopID_2_badOrders.items(), desc = "Finding the BADDEST Users for each Shop"):

		userBadnessCounter = Counter([x[0] for x in badOrders])
		maxUserBadness = max(userBadnessCounter.values())

		baddestUsers = sorted([int(k) for k, v in userBadnessCounter.items() if v == maxUserBadness])

		shopID_2_baddestUsers[int(shopID)] = baddestUsers



	# The correct answer should have 315 Shops w/ Order Brushing?
	countIdx = 0
	print("")
	for shopID, baddestUsers in sorted(shopID_2_baddestUsers.items()):

		if (countIdx >= len(shopID_2_baddestUsers) - 10):
			print("# {:<3d} \t\t Shop ID: {:15d} \t\t Baddest User(s): {}".format( countIdx + 1, shopID, ", ".join([str(x) for x in baddestUsers]) ))
		countIdx += 1

	print("\n(Possible) Score: {:.3f}".format( (18770 - countIdx) * 0.005 + countIdx ))



	# Save as CSV file
	with codecs.open("./submission.csv", 'w', encoding = 'utf-8', errors = 'ignore') as outFile:

		csvWriter = csv.writer(outFile, delimiter = ",", quotechar = '"')

		# Header row
		csvWriter.writerow(["shopid", "userid"])

		for shopID in sorted([int(x) for x in setShopIDs]):

			baddestUsers = shopID_2_baddestUsers[shopID]

			if (len(baddestUsers) == 0):

				csvWriter.writerow([str(shopID), "0"])

			else:

				baddestUsersStr = "&".join([str(x) for x in baddestUsers])
				csvWriter.writerow([str(shopID), baddestUsersStr])



	endTime = time.time()
	durationInSecs = endTime - startTime
	durationInMins = durationInSecs / 60
	print("\nTotal Time Taken: {:.2f} seconds ({:.2f} minutes)\n".format( durationInSecs, durationInMins ))


