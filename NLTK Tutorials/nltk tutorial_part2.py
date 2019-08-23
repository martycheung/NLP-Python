# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 22:01:15 2019

@author: martin.cheung
"""

import nltk
import random
from nltk.corpus import movie_reviews
from nltk.corpus import wordnet

# Wordnet - lexical database
syns = wordnet.synsets("program")

#synset
print(syns[0].name())
#word only
print(syns[0].lemmas()[0].name())
#definition
print(syns[0].definition())
#examples
print(syns[0].examples())
#synonyms/antonyms

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    print(syn)
    for l in syn.lemmas():
        print(l)
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

# Similarity
w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('car.n.01')
print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('cat.n.01')
print(w1.wup_similarity(w2))

