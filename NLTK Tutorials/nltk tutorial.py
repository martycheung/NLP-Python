# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 20:29:57 2019

@author: martin.cheung
"""

import nltk
from nltk.corpus import state_union
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import gutenberg

# Tokeniser
greeting = "Hi, my name is Martin. I work in a chocolate factory, and the boss, Mr. Joe comes along and says to me. You shouldn't do that"

print(nltk.sent_tokenize(greeting))
print(nltk.word_tokenize(greeting))

# Stop words
stop_words = set(stopwords.words("English"))

words = word_tokenize(greeting)

filtered_sentence = []

for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)

# Stemming
ps = PorterStemmer()
example_words = ["laugh","laughing","laughed","laughly","laughter"]

for w in example_words:
    print(ps.stem(w))

sentence = "It is important to by very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."

words = word_tokenize(sentence)

for w in word:
    print(ps.stem(w))

# Train Sentence Tokeniser using unsupervised ML 
train_text = state_union.raw("2005-GWBush.txt") 
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

# Part of Speech
# Chunking 
def process_content():
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()

# Chinking
def process_content():
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            chunkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT>{"""
            
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()
       
# Named Entity Recognition
def process_content():
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            
            namedEnt = nltk.ne_chunk(tagged)
            namedEnt.draw()
        
# Lemmatizing - similar to stemming, but gives real words
lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run",'v'))

sample = gutenberg.raw("bible-kjv.txt")
tok = sent_tokenize(sample)
for sent in range(5):
    print(tok[sent])
