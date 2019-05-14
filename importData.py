import os
import string
import quicksort
import insertionsort

os.chdir("C:\\Users\\yoonp\\independentCS\\MLTEST\\data\\")

data = open('train.tsv').read()

texts = []
labels = []

for i, line in enumerate(data.split("\n")):
	content = line.split("\t")
	#print(content)

	texts.append(content[2])
	labels.append(content[3])
	
def getTexts():
	return texts

def getLabels():
	return labels

