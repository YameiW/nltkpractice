# Categorizing and Tagging words
# 5.1 Using a Tagger
import nltk
text=nltk.word_tokenize('and now for something completely different')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('tagsets')
nltk.pos_tag(text)
nltk.help.upenn_tagset('RB')

text=nltk.Text(word.lower()for word in nltk.corpus.brown.words())
text.similar('woman')

# 5.2 Tagged Corpora
## representing tagged tokens
import nltk
tagged_tokens=nltk.tag.str2tuple('fly/NN')
tagged_tokens

## reading tagged corpora
nltk.corpus.brown.tagged_words()
# nltk.download('treebank')
nltk.corpus.brown.tagged_sents()

## A simplified part-of-speech tagset
from nltk.corpus import brown
brown_news_tagged=brown.tagged_words(categories='news',tagset='universal')
# nltk.download('universal_tagset')
tag_fd=nltk.FreqDist(tag for (word,tag) in brown_news_tagged)
tag_fd.keys()
tag_fd.plot(cumulative=True)

## Nouns
import nltk
word_tag_pairs=nltk.bigrams(brown_news_tagged)
list(word_tag_pairs)

list(nltk.FreqDist(a[1] for (a,b) in word_tag_pairs if b[1]=='N')) # EMPTY LIST ?

## Verbs
wsj=nltk.corpus.treebank.tagged_words(tagset='universal')
word_tag_fd=nltk.FreqDist(wsj)
[word+'/'+tag for (word,tag)in word_tag_fd if tag.startswith('V')]

## adjectives and adverbs
## unsimplified tags
## expliring tagged corpora

brown_learned_text=brown.words(categories='learned')
sorted(set(b for (a,b) in nltk.bigrams(brown_learned_text) if a=='often'))

brown_lrnd_tagged=brown.tagged_words(categories='learned',tagset='universal')
list(brown_lrnd_tagged)[1:10]
list(nltk.bigrams(brown_lrnd_tagged))[1:10]
tags=[b[1] for (a,b) in nltk.bigrams(brown_lrnd_tagged) if a[0]=='often']
fd=nltk.FreqDist(tags)
fd.tabulate()

def process(sentence):
    for (w1,t1),(w2,t2),(w3,t3) in nltk.trigrams(sentence):
        if (t1.startswith('V') and t2=='TO' and t3.startswith('V')):
            print (w1,w2,w3)

import nltk
for tagged_sent in nltk.corpus.brown.tagged_sents():
    process(tagged_sent)

# look for words that are highly ambiguous as to their pos tag
brown_news_tagged=nltk.corpus.brown.tagged_words(categories='news',tagset='universal')
data=nltk.ConditionalFreqDist((word.lower(),tag) for (word,tag) in brown_news_tagged)

for word in data.conditions():
    if len (data[word])>3:
        tags=data[word].keys()
        print(word,' '.join(tags))


# 5.3 Mapping words to properties using python dictionaries
## Indexing lists versus dictionaries
## Dictionaries in Python for mapping between arbitrary types

pos={}
pos['colorless']='ADJ' # key and value
pos['ideas']='N'
pos['sleep']='V'
pos['furiously']='ADV'
pos['ideas']

list(pos)
sorted(pos)
[w for w in pos if w.endswith('s')]

for word in sorted(pos):
    print (word+":",pos[word])

pos.keys()
pos.values()
pos.items()

for key,val in sorted(pos.items()):
    print(key+":",val)

pos['sleep']=['V','N']
pos

## Defining Dictionaries
pos={'colorless':'ADJ','ideas':'N','sleep':'V','furiously':'ADV'}
pos=dict(colorless='ADJ',ideas='N',sleep='V',furiously='ADV')
# keys must be immutable types, such as string and tuples.

## Default Dictionaries
import nltk
frequency=nltk.defaultdict(int)
frequency['colorless']=4
frequency['ideas']

pos=nltk.defaultdict(list)
pos['sleep']=['N','V']
pos['ideas']

pos=nltk.defaultdict(lambda:'N') # with a default value 'N'
pos['colorless']='ADJ'
pos['blog']
pos.items()

#
alice=nltk.corpus.gutenberg.words('carroll-alice.txt')
vocab=nltk.FreqDist(alice)
v1000=list(vocab)[1:1000]
mapping=nltk.defaultdict(lambda:'UNK') # UNK: out of vacabulary token
for v in v1000:
    mapping[v]=v

alice2=[mapping[v] for v in alice]
alice2[:100]                     

## Incrementally updating a dictionary
counts=nltk.defaultdict(int)

from nltk.corpus import brown
a=brown.tagged_words(categories='news')

for (word,tag) in brown.tagged_words(categories='news',tagset='universal'):
    counts[tag]+=1

counts['NOUN']
list(counts)

from operator import itemgetter
sorted(counts.items(),key=itemgetter(1),reverse=True)
[t for t,c in sorted (counts.items(),key=itemgetter(1),reverse=True)]

last_letters=nltk.defaultdict(list)
# nltk.download('words')
words=nltk.corpus.words.words('en')
for word in words:
    key=word[-2:]
    last_letters[key].append(word)

last_letters['ly']
last_letters['zy']

anagrams=nltk.defaultdict(list)

for word in words:
    key=''.join(sorted(word))
    anagrams[key].append(word)

anagrams['aeilnrt']

## Complex Key and Values
## Inventing a Dictionary
counts=nltk.defaultdict(int)
for word in nltk.corpus.gutenberg.words('milton-paradise.txt'):
    counts[word]+=1

[key for (key,value) in counts.items() if value==32]

pos={'colorless':'ADJ','ideas':'N','sleep':'V','furiously':'ADV'}
pos2=dict((value,key) for (key,value) in pos.items())
pos2['N']

pos.update({'cat':'N','scratch':'V','peacefully':'ADV','old':'ADJ'})
pos2=nltk.defaultdict(list)

for key, value in pos.items():
    pos2[value].append(key)

pos2['ADV']

pos2=nltk.Index((value,key) for (key,value) in pos.items())
pos2['ADJ']

# Automatic Tagging
from nltk.corpus import brown
brown_tagged_sents=brown.tagged_sents(categories='news')
brown_sents=brown.sents(categories='news')

## The default tagger
tags=[tag for (word,tag) in brown.tagged_words(categories='news')]
nltk.FreqDist(tags).max() # 'NN'

raw='I do not like green engs and ham, I do not like them Sam I am!'
tokens=nltk.word_tokenize(raw)
default_tagger=nltk.DefaultTagger('NN')
default_tagger.tag(tokens)

default_tagger.evaluate(brown_tagged_sents)

## the regular expression tagger
patterns=[(r'.ing$','VBG'),(r'.ed$','VBD'),(r'.es$','VBZ'),(r'.*ould$','MD'),(r'.*\'s$', 'NN$'),(r'.*s$','NNS'),
          (r'^-?[0-9]+(.[0-9]+)?$','CD'),(r'.*','NN')]

regexp_tagger=nltk.RegexpTagger(patterns)
regexp_tagger.tag(brown_sents[3])
regexp_tagger.evaluate(brown_tagged_sents)

## the lookup tagger
fd=nltk.FreqDist(brown.words(categories='news'))
cfd=nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
most_freq_words=list(fd.keys())[:100]
likely_tags=dict((word,cfd[word].max())for word in most_freq_words)
baseline_tagger=nltk.UnigramTagger(model=likely_tags)
baseline_tagger.evaluate(brown_tagged_sents)

sent=brown.sents(categories='news')[3]
baseline_tagger.tag(sent)

baseline_tagger= nltk.UnigramTagger(model=likely_tags,backoff=nltk.DefaultTagger('NN'))

def performance(cfd,wordlist):
    lt=dict((word,cfd[word].max())for word in wordlist)
    baseline_tagger=nltk.UnigramTagger(model=lt,backoff=nltk.DefaultTagger('NN'))
    return baseline_tagger.evaluate(brown.tagged_sents(categories='news'))


def display():
    import pylab
    words_by_freq=list(nltk.FreqDist(brown.words(categories='news')))
    cfd=nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
    sizes=2**pylab.arange(15)
    perfs=[performance(cfd,words_by_freq[:size]) for size in sizes]
    pylab.plot(sizes,perfs,'-bo')
    pylab.title('lookup tagger performance iwht varying model size')
    pylab.xlabel('model size')
    pylab.ylabel('performance')
    pylab.show()

display()   

# 5.5 N-gram tagging
from nltk.corpus import brown
brown_tagged_sents=brown.tagged_sents(categories='news')
brown_sents=brown.sents(categories='news')
unigram_tagger=nltk.UnigramTagger(brown_tagged_sents)
unigram_tagger.tag(brown_sents[2007])

## Separating the trainning and testing data
size=int(len(brown_tagged_sents)*0.9)
train_sents=brown_tagged_sents[:size]
test_sents=brown_tagged_sents[size:]
unigram_tagger=nltk.UnigramTagger(train_sents)
unigram_tagger.evaluate(test_sents)

## general N-gram tagging
bigram_tagger=nltk.BigramTagger(train_sents)
bigram_tagger.tag(brown_sents[2007])
unseen_sent=brown_sents[4203]
bigram_tagger.tag(unseen_sent)
bigram_tagger.evaluate(test_sents) # 0.102 sparse data problem

## combining taggers
t0=nltk.DefaultTagger('NN')
t1=nltk.UnigramTagger(train_sents,backoff=t0)
t2=nltk.BigramTagger(train_sents,backoff=t1)
t2.evaluate(test_sents)

## tagging unknown words

## storing taggers
import pickle
output=open('t2.plk','wb')
pickle.dump(t2,output,-1)
output.close()

input=open('t2.plk','rb')
tagger=pickle.load(input)
input.close()

text='''The borad's action shows that free enterprise
    is up against in our complex maze of regulatory laws.'''

tokens=text.split()
tagger.tag(tokens)

## performance limitation 
cfd=nltk.ConditionalFreqDist(
         ((x[1],y[1],z[0]),z[1])
         for sent in brown_tagged_sents
         for x,y,z in nltk.trigrams(sent))

ambiguous_contexts=[c for c in cfd.conditions() if len(cfd[c])>1]
sum(cfd[c].N() for c in ambiguous_contexts)/cfd.N()

test_tags=[tag for sent in brown.sents(categories='editorial')
               for (word,tag) in t2.tag(sent)]

gold_tags=[tag for (word,tag) in brown.tagged_words(categories='editorial')]

## Tagging Across sentence boundaries
brown_tagged_sents=brown.tagged_sents(categories='news')

size=int(len(brown_tagged_sents)*0.9)
train_sets=brown_tagged_sents[:size]
test_sets=brown_tagged_sents[size:]

t0=nltk.DefaultTagger('NN')
t1=nltk.UnigramTagger(train_sets,backoff=t0)
t2=nltk.BigramTagger(train_sets,backoff=t1)

t2.evaluate(test_sets)

# 5.6 Tranformation-based tagging: Brill tagging

# 5.7 How to determine the category of a word