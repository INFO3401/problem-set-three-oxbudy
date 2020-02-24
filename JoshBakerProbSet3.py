import pandas as pd
import matplotlib.pyplot as plt

#2
def loadAndCleanData(file):
	fileread = pd.read_csv(file)
	fileread.fillna(value=0, inplace=True)
	return fileread

#3
creditData = loadAndCleanData('creditData.csv')


#4
def computePDF(dataset, column):
	newplot = dataset[column].plot.kde()

headers = list(creditData.columns.values)

#4b

for i in headers:
	computePDF(creditData, i)
	plt.show(newplot)


#5 - comment out #4b for this one
def viewDistribution(dataframe, column):
	distplot = dataframe[column].hist(bins=5)
	distplot.set_title(column, fontsize=20)
	plt.show(distplot)
#6
def viewLogDistribution(dataframe, column):
	distlogplot = dataframe[column].hist(log=True, bins=6)
	distlogplot.set_title(column + " log", fontsize=20)
	plt.show(distlogplot)

#5 and 6

for i in headers:
	viewDistribution(creditData, i)
	viewLogDistribution(creditData, i)

#7
#Accidentally superated into 2 bins instead of 3, didn't realize
#till I had already done everything else
#RevolvingUtilizationOfUnsecuredLines Log = 0, 5000, 50000
#NumberOfTime30-59DaysPastDueNotWorse Log = 0, 10, 100
#DebtRatio Log = 0, 50000, 380000
#MonthlyIncome Log = 0, 250000, 3000000
#NumberOfOpenCreditLinesAndLoans Log = 0, 25, 60
#NumberOfTimes90DaysLate Log = 0, 8, 100
#NumberRealEstateLoansOrLines Log = 0, 11, 60
#NumberOfTime60-89DaysPastDueNotWorse Log = 0, 10, 100
#NumberOfDependents Log = 0, 4, 20

#8

probtable = {}

def computeDefaultRisk(column, feature, bin, df):
	count = 0.0
	between040 = 0.0

	for i, datapoint in df.iterrows():
		if datapoint[column] == 1:
			if datapoint[feature] >= bin[0] and datapoint[feature] <= bin[1]:
				count += 1
				between040 += 1
		else:
			if datapoint[feature] >= bin[0] and datapoint[feature] <= bin[1]:
				between040 += 1

	probability = count/between040
	try:
		if len(probtable[feature]) == 1:
			probtable[feature].append((bin, probability))
	except KeyError:
		probtable[feature] = [(bin, probability)]

	#print results
	print(feature + " [" + str(bin[0]) + "," + str(bin[1]) + "]: " + str(probability))


#9
computeDefaultRisk('SeriousDlqin2yrs', "age", [0,40], creditData)
computeDefaultRisk('SeriousDlqin2yrs', "age", [41,100], creditData)
computeDefaultRisk('SeriousDlqin2yrs',
"RevolvingUtilizationOfUnsecuredLines", [0,5000], creditData)
computeDefaultRisk('SeriousDlqin2yrs',
"RevolvingUtilizationOfUnsecuredLines", [5001,50000], creditData)
computeDefaultRisk('SeriousDlqin2yrs',
"NumberOfTime30-59DaysPastDueNotWorse", [0,10], creditData)
computeDefaultRisk('SeriousDlqin2yrs',
"NumberOfTime30-59DaysPastDueNotWorse", [11,100], creditData)
computeDefaultRisk('SeriousDlqin2yrs',
"DebtRatio", [0,50000], creditData)
computeDefaultRisk('SeriousDlqin2yrs',
"DebtRatio", [50001,380000], creditData)
computeDefaultRisk('SeriousDlqin2yrs',
"MonthlyIncome", [0,250000], creditData)
computeDefaultRisk('SeriousDlqin2yrs',
"MonthlyIncome", [250001,3000000], creditData)
computeDefaultRisk('SeriousDlqin2yrs',
"NumberOfOpenCreditLinesAndLoans", [0,25], creditData)
computeDefaultRisk('SeriousDlqin2yrs',
"NumberOfOpenCreditLinesAndLoans", [26,60], creditData)
computeDefaultRisk('SeriousDlqin2yrs',
"NumberOfTimes90DaysLate", [0,8], creditData)
computeDefaultRisk('SeriousDlqin2yrs',
"NumberOfTimes90DaysLate", [9,100], creditData)
computeDefaultRisk('SeriousDlqin2yrs',
"NumberRealEstateLoansOrLines", [0,11], creditData)
computeDefaultRisk('SeriousDlqin2yrs',
"NumberRealEstateLoansOrLines", [12,60], creditData)
computeDefaultRisk('SeriousDlqin2yrs',
"NumberOfTime60-89DaysPastDueNotWorse", [0,10], creditData)
computeDefaultRisk('SeriousDlqin2yrs',
"NumberOfTime60-89DaysPastDueNotWorse", [11,100], creditData)
computeDefaultRisk('SeriousDlqin2yrs',
"NumberOfDependents", [0,4], creditData)
computeDefaultRisk('SeriousDlqin2yrs',
"NumberOfDependents", [5,20], creditData)

#10
newLoans = loadAndCleanData('newLoans.csv')

#11

weights = {'age':0.025, 'NumberOfDependents':0.025, 'MonthlyIncome':0.1,
'DebtRatio':0.1, 'RevolvingUtilizationOfUnsecuredLines':0.1,
'NumberOfOpenCreditLinesAndLoans': 0.1, 'NumberRealEstateLoansOrLines':0.1,
'NumberOfTime30-59DaysPastDueNotWorse':0.15,
'NumberOfTime60-89DaysPastDueNotWorse':0.15, 'NumberOfTimes90DaysLate':0.15}

def predictDefaultRisk(data):
	finalprob = []
	colheaders = list(data.columns.values)
	for i, datapoint in data.iterrows():
		total = 0
		rowprobs = []
		colsgonethru = 0
		for headname in colheaders:
			if headname == "SeriousDlqin2yrs":
				colsgonethru += 1
				continue
			else:
				if datapoint[colsgonethru] >= probtable[headname][0][0][0] and datapoint[colsgonethru] <= probtable[headname][0][0][1]:
					probability = probtable[headname][0][1]
				elif datapoint[colsgonethru] >= probtable[headname][1][0][0] and datapoint[colsgonethru] <= probtable[headname][1][0][1]:
					probability = probtable[headname][1][1]

				weightedprob = weights[headname] * float(probability)
				rowprobs.append(weightedprob)
				colsgonethru += 1

		for i in rowprobs:
			total += float(i)
		finalprob.append(total)
	return finalprob


newLoans["SeriousDlqin2yrs"] = predictDefaultRisk(newLoans)
newLoans.to_csv('newLoansFull.csv', index=False)

computePDF(newLoans, "SeriousDlqin2yrs")
