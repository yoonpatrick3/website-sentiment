from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import pandas, xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import urlGetter, importData

from sklearn.datasets import load_iris
from sklearn import preprocessing



def train_model(classifier, feature_vector_train, label, feature_vector_valid):
    # fit the training dataset on the classifier
	classifier.fit(feature_vector_train, label)
	
    # predict the labels on validation dataset
	predictions = classifier.predict(feature_vector_valid)
	
	#print(classifier.predict_proba(feature_vector_valid))
    
	return predictions

def getPredictions(predictions, valid_y):
	return metrics.accuracy_score(predictions, valid_y)

'''
def findAvg(predTrain):
	websitePred = 0
	for numPred in predTrain:
		#do weighted average
		# 0 - 0.25, 1 - 0.2, 2 - 0,1, 3 - 0.2, 4 - 0.25
		weighted = 0
		if numPred == 1 or numPred == 3:
			weighted = 0.2
		elif numPred == 2:
			weighted = 0.1
		else:
			weighted = 0.25
		websitePred += (numPred*weighted)
	return websitePred
		

import os

os.chdir("C:\\Users\\yoonp\\independentCS\\MLTEST\\data\\trainingandtestdata")

data = open('training.csv').read()

labels, texts = [], []
for i, line in enumerate(data.split("\n")):
	content = line.split("\"")
	#print(content)
	if len(content)>=11:
		labels.append(content[1])
		texts.append(content[11])
'''

texts = importData.getTexts()
labels = importData.getLabels()

# creates a dataframe using texts and labels
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

# splits dataset into training and validation (unbiased evaluation of set in order to modify parameters within hidden neural laters) datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])


# TEST THIS OUT
'''
# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
'''
# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# Naive Bayes on ngram Level TF IDF Vectors
predvalid = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
accuracy = getPredictions(predvalid, valid_y)
print("NB, N-Gram Vectors: ", accuracy)

keepGoing = True

while keepGoing:
	
	websiteBool = True
	
	while websiteBool:
		try:
			input_reviews = urlGetter.extractUrl()
			websiteBool = False
		except:
			print("Error. Try Again")
	xreviews_tfidf_ngram = tfidf_vect_ngram.transform(input_reviews)
	predTrainString = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xreviews_tfidf_ngram)
	
	predTrain = [int(numeric_string) for numeric_string in predTrainString]
	
	print(predTrain)

	websitePred = 0
	
	'''
	iris = load_iris()
	print(iris.data.shape)
	X = iris.data[predTrain]
	
	standardized_X = preprocessing.scale(X)
	
	print(standardized_X)
	'''
	i = 0
	
	for numPred in predTrain: #standardized_X:
		websitePred += numPred
		if numPred != 2:
			print("Sentence: " + str(input_reviews[i]) + " | Sentiment: " + str(numPred))
		
		i +=1
	
		

	print("\nOverall Unweighted Sentiment: " + str(websitePred/len(predTrain)))
	'''
	weightedSent = findAvg(predTrain)
	print("Sum of Weighted Sentiment: " + str(weightedSent))
	print("Weighted Sentiment: " + str(weightedSent/len(predTrain)))
	'''
	#ERRROR CHECKING BROTHER
	str = input("Another website? [y/n]: ")
	if str!='y':
		break



