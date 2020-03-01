# writing structured programs
# 4.1 Back to the Basics
# assigment: the differences between modifying an object via an object reference and overwriting an object reference 
# equality: id() function to see the object reference 
# conditional: any() and all() functions

# 4.2 Sequences
t1="snark", # tuple has only one element
t2=() # an empty tuple

## operating on sequence types
import nltk
nltk.download()
raw='Red lorry, yellow lorry, red lorry, yellow lorry.'
text=nltk.word_tokenize(raw)
fdist=nltk.FreqDist(text)
for key in fdist:
    print (fdist[key],)
list(fdist)

a='I turned off the sepctroroute'
words=a.split()
words[2],words[3],words[4]=words[3],words[4],words[2] # tuples
words # rearrange the order of the list

# traditional way to move elment in the list
tmp=words[2]
words[2]=words[3]
words[3]=words[4]
words[4]=tmp
words

# zip() and enumerate()
words=['I','turned','off','the','spectroroute']
tags=['noun','verb','prep','det','noun']
a = list(zip(words,tags))
b=list(enumerate(words))
b

nltk.download('nps_chat')
text=nltk.corpus.nps_chat.words()
len(text)
cut=int(0.9*len(text))
training_data,test_data=text[:cut],text[cut:]
text==training_data+test_data # True
len(training_data)/len(test_data)


## Combining Different Sequence Types
words='I turned off the spectroroute'.split()
wordlens=[(len(word),word)for word in words]
wordlens.sort()
' '.join(w for (_,w) in wordlens)

# lists are mutable, whereas tupples are immutable
lexicon=[('the','det',['Di:','D@']),('off','prep',['Qf','O:f'])]
lexicon.sort()
lexicon[1]=('turned','VBD',['t3:nd','t3nd'])
del lexicon[0]
lexicon1=tuple(lexicon)
# lexicon1.sort() tuple object has no attribute 'sort'
# lexicon1[1]=('turned','VBD',['t3:nd','t3nd'])
# del lexicon1[1] tuple object does not support item deletion

## Generator Expressions
import nltk
text='''"When I use a word,", Humpty Dumpty said in rather a scornful tone,
"it means just what i choose it to mean - neither more nor less."'''
[w.lower() for w in nltk.word_tokenize(text)]
max([w.lower()for w in nltk.word_tokenize(text)]) 
max(w.lower()for w in nltk.word_tokenize(text)) # a generator expression.
min(w.lower()for w in nltk.word_tokenize(text))

# 4.3 Questions of Style
# Python Coding Style
# nltk.download('toolbox')
import re
from nltk.corpus import brown
rotokas_words=nltk.corpus.toolbox.words('rotokas.dic')
cv_word_pairs=[(cv,w) for w in rotokas_words
                      for cv in re.findall('[ptksvr][aeiou]',w)]

cfd=nltk.ConditionalFreqDist(
    (genre,word)
    for genre in brown.categories()
    for word in brown.words(categories=genre)
)

#if len(syllables)>4 and len(syllables[2])==3 and \                 \ to break a line
#    syllables[2][2] in [aeiou] and syllables[2][3]==syllables[1][3]:
#    process(syllable)

## Procedural Versus Declarative Style
fd=nltk.FreqDist(nltk.corpus.brown.words())
cumulative=0

for rank, word in enumerate(fd):
    cumulative += fd[word]*100/fd.N()
    print("%3d%6.2f%% %s"%(rank+1,cumulative,word))
    if cumulative >25:
       break

# find the longest word
text=nltk.corpus.gutenberg.words('milton-paradise.txt')
longest=''
for word in text:
    if len(word)>len(longest):
        longest=word

longest # find one longest word

maxlen=max(len(word) for word in text)
maxlen
[word for word in text if len(word)==maxlen] # find all longest words

## Some legitimate uses for counters
sent=['The','dog','gave','John','the','newspaper']
n=3
[sent[i:i+n] for i in range(len(sent)-n+1)]

a=nltk.ngrams(sent,3) # the function to do the n-gram
list(a)

# loop variable to build multidimensional structures
m,n=3,7
array=[[set() for i in range(n)] for j in range(m)]
array[2][5].add('Alice')
array

# 4.4 Functions:
## Function Inputs and Outputs
def repeat(msg,num):
    return ' '.join([msg]*num)
    
monty="Monty Python"
repeat(monty,3)

sent='The dog gives John the newspaper.'.split()
def my_sort1(mylist):
    mylist.sort() # modifies its argument, no return value

my_sort1(sent) 

def my_sort2(mylist):
    return sorted(mylist) # does not touch its argument, returns value

my_sort2(sent)

## Parameter Passing
def set_up(word,properties):
    word='lolcat'
    properties.append('noun')
    properties=5

w=''
p=[]
set_up(w,p)
w
p

# Variable scope
# Checking parameter types
def tag(word):
    assert isinstance(word,str),"argument to tag() must be a string"
    if word in ['a','the','all']:
        return 'det'
    else:
        return 'noun'
    
tag('the')

# Functional Decomposition
# Documenting Functions

## 4.5 Doing more with Functions
# Functions As Arguments
sent=['Take','care','of','the','sense',',','and','the','sounds','will','take','care','of','themselves','.']
def extract_property(prop):
    return[prop(word) for word in sent]

extract_property(len)

def last_letter(word):
    return word[-1]

extract_property(last_letter) # there is no parenthese after last_letter function, because it is treated as an object

extract_property(lambda w:w[-1]) # lambda expression

sorted(sent)
def cmp(x,y):
    if x==y:
        return 0
    elif x>y:
        return 1
    else:
        return -1


# Accumulative Functions 
def search1(substring,words):
    result=[]
    for word in words:
        if substring in word:
            result.append(word)
    return result

import nltk
print("search1:")
for item in search1('zz',nltk.corpus.brown.words()):
    print (item)

def search2(substring,words):
    for word in words:
        if substring in word:
            yield word

for item in search2('zz',nltk.corpus.brown.words()):
    print (item)

# recursion permutations to test syntax
def permutations(seq):
    if len(seq)<=1:
        yield seq
    else:
        for perm in permutations(seq[1:]):
            for i in range(len(perm)+1):
                yield perm[:i]+seq[0:1]+perm[i:]

list(permutations(['police','fish','buffalo']))

# higher-Order Functions
# higher ordier function applies a function to every item in a sequence
def is_content_word(word):
    return word.lower() not in ['a','of','the','and','will',',','.'] # Return True or False

sent=['Take','care','of','the','sense',',','and','the','sounds','will',\
'take','care','of','themselves','.']

# filter is a higher order function
list(filter(is_content_word,sent))
[w for w in sent if is_content_word(w)]

# a pair of equivalent examples that count the number of vowels in each word
[len([c for c in w if c.lower() in 'aeiou']) for w in sent]

# find out the average length of a sentence in the news
import nltk
lengths=map(len,nltk.corpus.brown.sents(categories='news'))
a=sum(lengths)
b=len(nltk.corpus.brown.sents(categories='news'))
a/b

lengths=[len(w) for w in nltk.corpus.brown.sents(categories='news')]
sum(lengths)/len(lengths)

## Named Arguments
def repeat(msg='<empty>',num=1):
    return msg*num

repeat(num=3)
repeat(msg='Alice')
repeat(num=3,msg='Alice')

def generic(*args,**kwargs):
    print(args)
    print(kwargs)

generic(1,'Africa swallow',monty='python')

song=[['four','calling','birds'],['three','French','hens'],['two','turtle','doves']]
list(zip(song[0],song[1],song[2]))
list(zip(*song))

def freq_words(file,min=1,num=10):
    text=open(file).read()
    tokens=nltk.word_tokenize(text)
    freqdist=nltk.FreqDist(t for t in tokens if len(t)>=min)
    return freqdist.keys()[:num]

# 4.6 Program Development
## Structure of a Python Module
## Multimodule Programs
## Sources of Error

# 4.7 Algorithm Design
## Recursion
## Space-Time Trade-offs
## Dynamic programming
# iterative
def virahanka1(n):
    if n==0:
        return['']
    elif n==1:
        return ['S']
    else:
        s=['S'+prosody for prosody in virahanka1(n-1)]
        l=['L'+prosody for prosody in virahanka1(n-2)]
        return s+l

def virahanka2(n):
    lookup=[[''],['S']]
    for i in range(n-1):
        s=['S'+prosody for prosody in lookup[i+1]]
        l=['L'+prosody for prosody in lookup[i]]
        lookup.append(s+l)
    return lookup[n]

# 4.8 A sample of Python Libraries
































