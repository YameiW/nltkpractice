# 6.1 supervised classification
## general idenfication
def gender_features(word):
    return{'last_letter':word[-1]}

gender_features('Shrek')

import nltk
# nltk.download('names')
from nltk.corpus import names
import random
names=([(name,'male') for name in names.words('male.txt')]+
       [(name,'female')for name in names.words('female.txt')])

random.shuffle(names)

featuresets=[(gender_features(n),g) for (n,g) in names]
train_set,test_set=featuresets[500:],featuresets[:500]
classifier=nltk.NaiveBayesClassifier.train(train_set)

classifier.classify(gender_features('Neo'))
classifier.classify(gender_features('Trinity'))

print(nltk.classify.accuracy(classifier,test_set)) # 0.774

classifier.show_most_informative_features(5)

## choosing the right features

def gender_feature2(name):
    features={}
    features['firstletter']=name[0].lower()
    features['lastletter']=name[-1].lower()
    for letter in 'abcdefghijklmnopqrstuvwxyz':
        features['count(%s)'% letter] = name.lower().count(letter)
        features['has(%s)'% letter]=(letter in name.lower())
    return features

gender_feature2('John')

featuresets=[(gender_feature2(n),g) for (n,g) in names]
train_set,test_set=featuresets[:500],featuresets[500:]
classifier=nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier,test_set)) # 0.745

#development set (training set and dev-test set), and dev-test set is for error analysis
train_names=names[1500:]
devtest_names=names[500:1500]
test_names=names[:500]

train_set=[(gender_features(n),g) for (n,g) in train_names]
devtest_set=[(gender_features(n),g) for (n,g) in devtest_names]
test_set=[(gender_features(n),g) for (n,g) in test_names]

classifier=nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier,devtest_set)) # 0.758

error=[]

for (name,tag) in devtest_names:
    guess=classifier.classify(gender_feature2(name))
    if guess != tag:
        error.append((tag,guess,name))

for (tag,guess,name) in sorted(error):
    print('correct=%-8s guess=%-8s name=%-30s' %(tag,guess,name))

def gender_features(word):
    return{'suffix1':word[-1:],
           'suffix2':word[-2:]}

train_set=[(gender_features(n),g) for (n,g) in train_names]
devtest_set=[(gender_features(n),g) for (n,g) in devtest_names]
test_set=[(gender_features(n),g) for (n,g) in test_names]

classifier=nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier,devtest_set)) #0.769

print(nltk.classify.accuracy(classifier,test_set)) #0.79

## Document Classification
from nltk.corpus import movie_reviews
import random
# nltk.download('movie_reviews')

documents=[(list(movie_reviews.words(fileid)),category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

import nltk
all_words=nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features=list(all_words.keys())[:2000]

def document_features(document):
    document_words=set(document)
    features={}
    for word in word_features:
        features['contains(%s)'% word]=(word in document_words)
    return features

print(document_features(movie_reviews.words('pos/cv957_8737.txt')))

featuresets=[(document_features(d),c) for (d,c) in documents]
train_set,test_set=featuresets[100:],featuresets[:100]
classifier=nltk.NaiveBayesClassifier.train(train_set)

print(nltk.classify.accuracy(classifier,test_set)) # 0.81
classifier.show_most_informative_features(5)

## POS tagging
from nltk.corpus import brown
suffix_fdist=nltk.FreqDist()
for word in brown.words():
    word=word.lower()
    suffix_fdist[word[-1:]]+=1
    suffix_fdist[word[-2:]]+=1
    suffix_fdist[word[-3:]]+=1

common_suffix=[]

for suffix in suffix_fdist.most_common(100):
    common_suffix.append(suffix[0])

print(common_suffix)

def pos_features(word):
    features={}
    for suffix in common_suffix:
        features['endswith(%s)'%suffix]=word.lower().endswith(suffix)
    return features

tagged_words=brown.tagged_words(categories='news')
featuresets=[(pos_features(n),g) for (n,g) in tagged_words]

size=int(len(featuresets)*0.1)
train_set,test_set=featuresets[size:],featuresets[:size]

classifier=nltk.DecisionTreeClassifier.train(train_set)
nltk.classify.accuracy(classifier,test_set) # 0.627

classifier.classify(pos_features('cats'))

print(classifier.pseudocode(depth=4))

## Exploiting Context
from nltk.corpus import brown

def pos_features(sentence,i):
    features={'suffix(1)':sentence[i][-1:],
              'suffix(2)':sentence[i][-2:],
              'suffix(3)':sentence[i][-3:]}
    if i ==0:
        features['prev-word']='<START>'
    else:
        features['prev-word']=sentence[i-1]
    return features

brown.sents()[0]
pos_features(brown.sents()[0],8)

tagged_sents=brown.tagged_sents(categories='news')
featuresets=[]

for tagged_sent in tagged_sents:
    untagged_sent=nltk.tag.untag(tagged_sent)
    for i, (word, tag) in enumerate(tagged_sent):
        featuresets.append((pos_features(untagged_sent,i),tag))

size=int(len(featuresets)*0.1)
train_set,test_set=featuresets[size:],featuresets[:size]
classifier=nltk.NaiveBayesClassifier.train(train_set)

nltk.classify.accuracy(classifier,test_set) # 0.789

## Sequence classification

def pos_features(sentence,i,history):
    features={'suffix(1)':sentence[i][-1:],
              'suffix(2)':sentence[i][-2:],
              'suffix(3)':sentence[i][-3:]}
    
    if i==0:
        features['prev-word']='<START>'
        features['prev-tag']='<START>'
    else:
        features['prev-word']=sentence[i-1]
        features['prev-tag']=history[i-1]
    return features

class ConsecutivePosTagger(nltk.TaggerI): # A sequence classifier
    def __init__(self,train_sents):
        train_set=[]
        for tagged_sent in train_sents:
            untagged_sent=nltk.tag.untag(tagged_sent)
            history=[]
            for i,(word,tag) in enumerate(tagged_sent):
                featureset=pos_features(untagged_sent,i,history)
                train_set.append((featureset,tag))
                history.append(tag)
        self.classifier=nltk.NaiveBayesClassifier.train(train_set)

    def tag(self,sentence):
        history=[]
        for i,word in enumerate(sentence):
            featureset=pos_features(sentence,i,history)
            tag=self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence,history)
    
tagged_sents=brown.tagged_sents(categories='news')   
size=int(len(tagged_sents)*0.1)
train_sents,test_sents= tagged_sents[size:],tagged_sents[size:]
tagger=ConsecutivePosTagger(train_sents)
print(tagger.evaluate(test_sents))

## Other methods for Sequence Classification

# 6.2 Further Examples of Supervised Classification

## Sentence Segmentation
import nltk.corpus
sents=nltk.corpus.treebank_raw.sents()
tokens=[]
boundaries=set()
offset=0

for sent in sents:
    tokens.extend(sent)
    offset+=len(sent)
    boundaries.add(offset-1)

def punct_features(tokens,i):
    return{'next-word-capitalized':tokens[i+1][0].isupper(),
           'prevword':tokens[i-1].lower(),
           'punct':tokens[i],
           'prev-word-is-one-char':len(tokens[i-1])==1}

featuresets=[(punct_features(tokens,i),(i in boundaries))
             for i in range(1,len(tokens)-1)
             if tokens[i] in '.?!']

size=int(len(featuresets)*0.1)
train_set,test_set=featuresets[size:],featuresets[:size]
classifier=nltk.NaiveBayesClassifier.train(train_set)
nltk.classify.accuracy(classifier,test_set) # 0.93

def segment_sentences(words):
    start=0
    sents=[]
    for i,word in words:
        if word in '.?!' and classifier.classify(word,i)==True:
            sents.append(words[start:i+1])
            start=i+1
        if start<len(words):
            sents.append(words[start:])
        

## Identifying Dialogue Act Types
import nltk
posts=nltk.corpus.nps_chat.xml_posts()[:10000]


def dialogue_act_features(post):
    features={}
    for word in nltk.word_tokenize(post):
        features['contains(%s)'% word.lower()]=True
        return features
    
featuresets=[(dialogue_act_features(post.text),post.get('class'))
            for post in posts]

size=int(len(featuresets)*0.1)
train_set,test_set=featuresets[size:],featuresets[:size]
classifier=nltk.NaiveBayesClassifier.train(train_set)
print(nltk.classify.accuracy(classifier,test_set)) #0.749

## Recognizing Textual Entailment (RTE)

def rte_features(rtepair):
    extractor=nltk.RTEFeatureExtractor(rtepair)
    features={}
    features['word_overlap']=len(extractor.overlap('word'))
    features['word_hyp_extra']=len(extractor.hyp_extra('word'))
    features['ne_overlap']=len(extractor.overlap('ne'))
    features['ne_hyp_extra']=len(extractor.hyp_extra('ne'))
    return features

rtepair=nltk.corpus.rte.pairs(['rte3_dev.xml'])[33]
extractor=nltk.RTEFeatureExtractor(rtepair)

print(extractor.text_words)

print(extractor.hyp_words)

print(extractor.overlap('word'))

print(extractor.hyp_extra('word'))

print(extractor.overlap('ne'))

print(extractor.hyp_extra('ne'))


## Scaling up to large Datasets

# 6.3 Evaluation
## the test set
# The more similar these two datasets are, the less confident we can be that evaluation results will generalize to other datasets.
import random
from nltk.corpus import brown
tagged_sents=list(brown.tagged_sents(categories='news'))
random.shuffle(tagged_sents)
size=int(len(tagged_sents)*0.1)
train_set,test_set=tagged_sents[size:],tagged_sents[:size]

# a better way
file_ids=brown.fileids(categories='news')
size=int(len(file_ids)*0.1)
train_set=brown.tagged_sents(file_ids[size:]) # different documents under the same genre
test_set=brown.tagged)sents(file_ids[:size])

train_set=brown.tagged_sents(categories='news') # different genres
test_set=brown.tagged_sents(categoiries='fiction')

## Accuracy
## Precision and Recall
## Confusion Matrices
def tag_list(tagged_sents):
    return [tag for sent in tagged_sents for (word,tag) in sent]

def apply_tagger(tagger,corpus):
    return[tagger.tag(nltk.tag.untag(sent))for sent in corpus]

input=open('t2.plk','rb')
import pickle
t2=pickle.load(input)

gold=tag_list(brown.tagged_sents(categories='editorial'))
test=tag_list(apply_tagger(t2,brown.tagged_sents(categories='editorial')))
cm=nltk.ConfusionMatrix(gold,test)
print(cm)

## Cross-Validation

# 6.4 Decision Trees
## Entropy and Information Gain
import math
def entropy(labels):
    freqdist=nltk.FreqDist(labels)
    probs=[freqdist.freq(l) for l in nltk.FreqDist(labels)]
    return -sum([p*math.log(p,2) for p in probs])

print(entropy(['male','female','male','female'])) #1
print(entropy(['male','male','male','male'])) #0


# 6.5 Naive Bayes Classifiers
## Underlying probabilistic model
## Zero Counts and Smoothing
## Non-Binary Features
## Naivete of Independence 
## The Cause of Double-Counting

# 6.6 Maximum Entropy Classifiers
# 6.7 Modeling linguistic patterns




















