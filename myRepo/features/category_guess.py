import argparse
import string

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

class Analyzer:
    def __init__(self, word,length):
        self.word = word
        self.length = length
    
    def __call__(self, feature_string):
        #print feature_string
        #print
        #feats = feature_string.split('$')
        
        if self.word:
            yield feature_string
        
        if self.length:
            yield feats[1]
        '''
        if self.after:
            for ii in [x for x in feats if x.startswith("A:")]:
                yield ii
        if self.before:
            for ii in [x for x in feats if x.startswith("B:")]:
                yield ii
        if self.prev:
            for ii in [x for x in feats if x.startswith("P:")]:
                yield ii
        if self.next:
            for ii in [x for x in feats if x.startswith("N:")]:
                yield ii
        if self.dict:
            for ii in [x for x in feats if x.startswith("D:")]:
                yield ii
        if self.char:
            for ii in [x for x in feats if x.startswith("C:")]:
                yield ii
        '''


class Featurizer:
    def __init__(self):
        analyzer = Analyzer(True,False)
        #mostCommon = "boar theology discussion solomon ritual festival gods mythological built harbor feature mountain fault bridge red neutral parameter resistance physical classical physicist wavelength scattering quark physics background painter canvas architect foreground painting piano opera composer name russia convention battles byzantine execution successor capture foreign legislation empire radio clouds classification mass atmosphere object objects sun galaxies astronomical letter cursed decrease faith analyzes sermon study output kings thinker authors decides poems literature novel play stories literary playwright novels subset proof kind constructs methods abelian euler curve algebraic mathematician team season stanley code strip starring movies oscar lorentz laws largest above down constellation computer angle mechanics increase selection syndrome assay membrane cellular enzyme humans plants chromosome genes reaction reacts obtained molecular measured exponential molar presence chemistry chemist rock low boundary mantle period hot zone discontinuity mineral material"
        #mostCommon = list(set(mostCommon.split(' ')))
        stopwords = "the of and to a in for is on that by this with i you it not or be are from at as your all have new more an was we will home can us about if page my has search free but our one other do no information time they site he up may what which their news out use any there see only so his when contact here business who web also now help get pm view online first am been would how were me services some these click its like service than find"
        stopwords = stopwords.split()
        stopwords = text.ENGLISH_STOP_WORDS
        self.vectorizer = CountVectorizer(ngram_range=(1,2),stop_words=stopwords)#,max_df=0.4)
    
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

def example(question,labels,train=True):
    #word = sentence[position][0]
    #ex = word
    ex = question['text']
    
    
    #sent = ex.split('.')
    #length = len(sent)-1
    #ex = ex.split(' ')
    #ex = ex[0]
    #ex = ex[len(ex)-2]
    #ex = ex[0]
    #tag = normalize_tags(sentence[position][1])
    if train:
        target = labels.index(question['cat'])
    else:
        target = -1
        #if tag in kTAGSET:
        #target = kTAGSET.index(tag)
        #else:
        #target = None
        
        #ex+='$'
#ex+=str(length)
    
    return ex, target


def all_examples(examples,limit,categories,train=True):
    ex_num = 0
    for ii in examples:
        ex_num += 1
        if limit > 0 and ex_num > limit:
            break
    
        ex, tgt = example(ii,labels,train)
        
        yield ex, tgt



def accuracy(classifier, x, y, examples,categories):
    predictions = classifier.predict(x)
    cm = confusion_matrix(y, predictions)
    
    print("Accuracy: %f" % accuracy_score(y, predictions))
    
    
    print("\t".join(categories))
    for ii in cm:
        print("\t".join(str(x) for x in ii))
    
    errors = defaultdict(int)

#print enumerate(examples)

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
    parser.add_argument('--limit', default=-1, type=int,
                    help="How many sentences to use")
    
    flags = parser.parse_args()

    # Cast to list to keep it all in memory
    train = list(DictReader(open("train.csv", 'r')))
    totalTrain = len(train)
    shuffle(train)
    test = list(DictReader(open("test.csv", 'r')))
    
    feat = Featurizer()

    labels = []
    for line in train:
        if not line['cat'] in labels:
            labels.append(line['cat'])
    #print labels
    print

    x_train = feat.train_feature(ex for ex, tgt in
                                 all_examples(train[0:flags.limit],flags.limit,labels)) #all the words


    x_test = feat.test_feature(ex for ex, tgt in
                           all_examples(test,flags.limit,labels,train=False))

    y_train = array(list(tgt for ex, tgt in
                         all_examples(train[0:flags.limit],flags.limit,labels)))

    corr_x = feat.test_feature(ex for ex, tgt in
                                    all_examples(train[flags.limit:totalTrain],flags.limit,labels,train=True))

    corr_y = array(list(tgt for ex, tgt in
                        all_examples(train[flags.limit:totalTrain],flags.limit,labels)))


    #print y_train

    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)

    feat.show_top10(lr, labels)

    print("TRAIN\n-------------------------")
    accuracy(lr, x_train, y_train,all_examples(train[0:flags.limit],flags.limit,labels),labels)

    print("CORR\n-------------------------")
    accuracy(lr, corr_x, corr_y,all_examples(train[flags.limit:totalTrain],flags.limit,labels),labels)

    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["id", "cat"])
    o.writeheader()
    for ii, pp in zip([x['id'] for x in test], predictions):
        d = {'id': ii, 'cat': labels[pp]}
        o.writerow(d)
