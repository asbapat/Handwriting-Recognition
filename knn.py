import random
import csv
import math
import operator
from csv import reader
import Image

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'r') as csvfile:
	    lines = csv.reader(csvfile)
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(64):
	            dataset[x][y] = float(dataset[x][y])
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])
	# print("trainingSet here:",trainingSet);  
	print("trainingSet here:",testSet);            

def calcEuclidDistance(array1, array2, length):
	distance = 0
	for x in range(length):
		distance += pow((array1[x] - array2[x]), 2)
	return math.sqrt(distance)  

def getKNN(trainingSet, testInstance, k):
	distances =[]
	length = len(testInstance) -1;
	for x in range(len(trainingSet)):
		dist = calcEuclidDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors


def PredictValues(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]

def calcAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] is predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def main():
	
	trainingSet=[]
	testSet=[]
	filename_train = 'handWritting_Trian.csv'
	loadDataset(filename_train,1,trainingSet,testSet)
	filename_test = 'optdigits_test.csv'
	loadDataset(filename_test,0,trainingSet,testSet)

	predictions=[]
	k = 10
	for x in range(len(testSet)):
		neighbors = getKNN(trainingSet, testSet[x], k)
		result = PredictValues(neighbors)
		predictions.append(result)
	accuracy = calcAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')
	
main()			          