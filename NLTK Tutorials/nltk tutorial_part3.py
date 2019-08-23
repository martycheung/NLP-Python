# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 23:20:38 2019

@author: martin.cheung
"""

import nltk
import random
from nltk.corpus import movie_reviews
import pickle

# Text Classification - bag of words
documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)),category))

print(documents[1])

random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower)
    
all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))
print(all_words["hello"]) # how many times hello in corpus

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

print(find_features(movie_reviews.words('neg/cv000_29416.txt')))

featuresets = [(find_features(review),category) for (review,category) in documents]

train_set = featuresets[:1900]
test_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(train_set)

print("Naive Bayes Algo Acc %: ", (nltk.classify.accuracy(classifier,test_set))*100)
classifier.show_most_informative_features(15)

# Save classifier using Pickle
save_classifier = open("naivebayes_test.pickle","wb")
pickle.dump(classifier,save_classifier)
save_classifier.close()

classifier_f = open("naivebayes_test.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()




