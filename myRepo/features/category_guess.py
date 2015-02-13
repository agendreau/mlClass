#Alex Gendreau
#Machine Learning
#Homework 3: Feature Engineering


import argparse
import string
import re

from csv import DictReader, DictWriter
from random import shuffle

import numpy as np
from numpy import array
from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import text

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk.util import ngrams

import operator

#analyzer that determines which features to use
class Analyzer:
    def __init__(self, word,cap,length):
        self.word = word
        self.cap = cap
        self.length = length
    
    def __call__(self, feature_list):
       
        
        if self.cap:
            cap = feature_list.pop()
            yield cap
        if self.length:
            l = feature_list.pop()
            yield l
        if self.word:
            words = [x.strip() for x in feature_list]
            for i in range(len(words)):
                j = i+1
                #k = i+2
                if j<len(words)-1:
                    yield words[i]+words[j]
                #if k<len(words)-2:
                    #yield words[i]+words[i+1]+words[i+2]
                yield words[i]



class Featurizer:
    def __init__(self):
        analyzer = Analyzer(True,True,True)
        
        self.vectorizer = CountVectorizer(analyzer=analyzer)
        #Other stuff I messed around with
        #,ngram_range=(2,2)) #,ngram_range=(1,3),stop_words=stopwords)
        #,max_df=0.4)
        #self.vectorizer = VarianceThreshold(threshold=(.8 * (1 - .8)))

    def train_feature(self, examples):
        test = self.vectorizer.fit_transform(examples)
        #print self.vectorizer.get_feature_names()
        #print self.vectorizer.inverse_transform(test)
        return test

    def test_feature(self, examples):
        return self.vectorizer.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        for i, category in enumerate(categories):
            top10 = np.argsort(classifier.coef_[i])[-10:]
            print("%s: %s" % (category, " ".join(feature_names[top10])))

#Modified from features.py.  All the possible features of an question
def example(question,labels,train=True):
    #word = sentence[position][0]
    #ex = word
    #Too many tokens potentially over fitting, how to reduce tokens but maintain results?
    #Remove words and the padding from words
    #Remove common words (i.e. by using stop words)
    #Remove numbers
    myStop = "the of and to a in for is on that by this with i you it not or be are from at as your all have new more an was we will home can us about if page my has search free but our one other do no information time they site he up may what which their news out use any there see only so his when contact here business who web also now help get pm view online first am been would how were me services some these click its like service than find"
    myStop = set(myStop.split())
    stopwords = text.ENGLISH_STOP_WORDS | myStop
    ex = question['text']
    totalCap = 0
    words = ex.split()
    
    toRemove = string.punctuation+','+'[\W_]'+';'
    endings ='[ed][ing][\'s]'
    words = [w.strip(toRemove) for w in words] #remove punctuation
    words = [w.rstrip(endings) for w in words] #remove padding
    words = [w.strip('\\n') for w in words] #for some reason weird newlines occurred
    
    
    for w in words: #how many capital letters
        if w.istitle():
            totalCap+=1

    wordsToUse = []
    for w in words:
        if w not in stopwords and len(w)>0:
            wordsToUse.append(w)
    length = len(ex.split('.'))-1 #number of sentences
    wordsToUse.append(length)
    wordsToUse.append(totalCap-length) #don't count first word of sentences
    if train:
        target = labels.index(question['cat'])
    else:
        target = -1 #unsurpervised, we don't know the info about our test set
    
    return wordsToUse, target


def all_examples(examples,limit,categories,train=True): #from feature.py all examples
    ex_num = 0
    for ii in examples:
        ex_num += 1
        if limit > 0 and ex_num > limit:
            break
    
        ex, tgt = example(ii,labels,train)
        
        yield ex, tgt


#measure the accuracy of the classifier using the confusion matrix
def accuracy(classifier, x, y, examples,categories):
    predictions = classifier.predict(x)
    cm = confusion_matrix(y, predictions)
    
    print("Accuracy: %f" % accuracy_score(y, predictions))
    
    
    print("\t".join(categories))
    for ii in cm:
        print("\t".join(str(x) for x in ii))
    
    errors = defaultdict(int)

    for ii, ex_tuple in enumerate(examples):
        #print ex_tuple
        ex, tgt = ex_tuple
        if tgt != predictions[ii]:
            errors[(categories[tgt], categories[predictions[ii]])] += 1

    print "here"
    for ww, cc in sorted(errors.items(), key=operator.itemgetter(1),reverse=True)[:10]:
        print("%s\t%i" % (ww, cc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    #argument determines how much of training set to use for actual training and
    #how much to save for cross validation
    parser.add_argument('--limit', default=-1, type=int,
                        help="How many sentences to use")
    parser.add_argument('--compute', default=False, action='store_true',
                        help="Actually Predict Test")
    
    
    flags = parser.parse_args()
    

    # Cast to list to keep it all in memory
    train = list(DictReader(open("train.csv", 'r')))
    totalTrain = len(train)
    shuffle(train) #randomize test data so training and validation have similar distributions
    test = list(DictReader(open("test.csv", 'r')))
    
    if(flags.compute):
        flags.limit = totalTrain
    
    feat = Featurizer()

    labels = []
    for line in train:
        if not line['cat'] in labels:
            labels.append(line['cat'])


    x_train = feat.train_feature(ex for ex, tgt in
                                 all_examples(train[0:flags.limit],flags.limit,labels)) #all the words


    x_test = feat.test_feature(ex for ex, tgt in
                           all_examples(test,flags.limit,labels,train=False))

    y_train = array(list(tgt for ex, tgt in
                         all_examples(train[0:flags.limit],flags.limit,labels)))

#set up cross validation
    if not flags.compute:
        corr_x = feat.test_feature(ex for ex, tgt in
                                    all_examples(train[flags.limit:totalTrain],flags.limit,labels,train=True))

        corr_y = array(list(tgt for ex, tgt in
                        all_examples(train[flags.limit:totalTrain],flags.limit,labels)))




    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    feat.show_top10(lr, labels)

#Training set and Cross Validation Set Accuracy
    print("TRAIN\n-------------------------")
    accuracy(lr, x_train, y_train,all_examples(train[0:flags.limit],flags.limit,labels),labels)
    if not flags.compute:
        print("CORR\n-------------------------")
        accuracy(lr, corr_x, corr_y,all_examples(train[flags.limit:totalTrain],flags.limit,labels),labels)

    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "cat"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'cat': labels[pp]}
        o.writerow(d)
