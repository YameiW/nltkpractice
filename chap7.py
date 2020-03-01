# 7.1 Information Extraction
## Information Extraction Architecture
# a function: default sentence segmentation, word tokenizer, and part of speech tagger
import nltk
def ie_preprocess(document):
    sentences=nltk.sent_tokenize(document)
    sentences=[nltk.word_tokenize(sent) for sent in sentences]
    sentences=[nltk.pos_tag(sent) for sent in sentences]

# 7.2 Chunking
## Noun phrase Chuncking
sentence=[('the','DT'),('little','JJ'),('yellow','JJ'),('dog','NN'),('barked','VBD'),
          ('at','IN'),('the','DT'),('cat','NN')]

grammar='NP:{<DT>?<JJ>*<NN>}'
cp=nltk.RegexpParser(grammar)
result=cp.parse(sentence)
print(result)
result.draw()

## Tag Patterns
## Chunking with Regular Expressions
grammar=r"""
     NP:{<DT|PP\$>?<JJ>*<NN>}
        {<NNP>+}
"""
cp=nltk.RegexpParser(grammar)
sentence=[('Rapunzel','NNP'),('let','VBD'),('down','RP'),('her','PP$'),('long','JJ'),
          ('golden','JJ'),('hair','NN')]
print(cp.parse(sentence))
cp.parse(sentence).draw()

nouns=[('money','NN'),('market','NN'),('fund','NN')]
grammar='NP:{<NN><NN>}' # chunk two consecutive nouns

# the leftmost match takes precedence

cp=nltk.RegexpParser(grammar)
cp.parse(nouns).draw()


## Exploring Text Corpora

cp=nltk.RegexpParser('CHUNK:{<V.*><TO><V.*>}')
brown=nltk.corpus.brown

for sent in brown.tagged_sents():
    tree=cp.parse(sent)
    for subtree in tree.subtrees():
        if subtree.label()=='CHUNK': print(subtree) 

## Chinking
# simple chinker

grammar=r"""
    NP:
      {<.*>+}      # chunk everything
      }<VBD|IN>+{   # chink sequences of VBD and IN
"""
sentence=[('the','DT'),('little','JJ'),('yellow','JJ'),('dog','NN'),('barked','VBD'),
          ('at','IN'),('the','DT'),('cat','NN')]

cp=nltk.RegexpParser(grammar)
print(cp.parse(sentence))

## Representing Chunks: Tags Versus Trees

# 7.3 Developing and Evaluating Chunkers
## Reading IOB format and the CoNLL-2000 Chunking Corpus
from nltk.corpus import conll2000
print(conll2000.chunked_sents('train.txt')[99])

print(conll2000.chunked_sents('train.txt',chunk_types=['NP'])[99])

## Simple Evaluation and Baselines
from nltk.corpus import conll2000
cp=nltk.RegexpParser("") # to create no chunks 
test_sents=conll2000.chunked_sents('test.txt',chunk_types=['NP'])
print(cp.evaluate(test_sents))

grammar=r"NP:{<[CDJNP].*>+}"
cp=nltk.RegexpParser(grammar)
print(cp.evaluate(test_sents))

import nltk
class UnigramChunker(nltk.ChunkParserI):
    def __init__(self,train_sents):
        train_data=[[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                    for sent in train_sents]
        self.tagger=nltk.UnigramTagger(train_data)

    def parse(self,sentence):
        pos_tags=[pos for (word,pos) in sentence]
        tagged_pos_tags=self.tagger.tag(pos_tags)
        chunktags=[chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags=[(word,pos,chunktag) for ((word,pos),chunktag) in zip(sentence,chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)

from nltk.corpus import conll2000
test_sents=conll2000.chunked_sents('test.txt',chunk_types=['NP'])
train_sents=conll2000.chunked_sents('train.txt',chunk_types=['NP'])
unigram_chunker=UnigramChunker(train_sents)
print(unigram_chunker.evaluate(test_sents)) # NO RESULT P.273

postags=sorted((set(pos for sent in train_sents
                for (word,pos) in sent.leaves())))

print(unigram_chunker.tagger.tag(postags))


## Trainning Classifier-Based Chunkers

def npchun_features(sentence,i,history):
    word,pos=sentence[i]
    return {'pos':pos}

class ConsecutiveNPChunkTagger(nltk.TaggerI):

    def __init__(self,train_sents):
        train_set=[]
        for tagged_sent in train_sents:
            untagged_sent=nltk.tag.untag(tagged_sent)
            history=[]
            for i,(word,tag) in enumerate(tagged_sent):
                featureset=npchunk_features(untagged_sent,i,history)
                train_set.append((featureset,tag))
                history.append(tag)
        self.classifier=nltk.MaxentClassifier.train(train_set,algorithm='megam',trace=0)

    def tag(self,sentence):
        history=[]
        for i,word in enumerate(sentence):
            featureset=npchunk_features(sentence,i,history)
            tag=self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence,history)

class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self,train_sents):
        tagged_sents=[[((w,t),c) for (w,t,c) in nltk.chunk.tree2conlltags(sent)]
                                 for sent in train_sents]
        self.tagger=ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self,sentence):
        tagged_sents=self.tagger.tag(sentence)
        conlltags=[(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)


def npchun_features(sentence,i,history):
    word,pos=sentence[i]
    return {'pos':pos}

chunker=ConsecutiveNPChunkTagger(train_sents2) 
print(chunker.evaluate(test_sents))


# 7.4 Recursion in Linguistic Structure
## Building nested structure with cascaded chunckers
import nltk
grammar=r"""
  NP:{<DT|JJ|NN.*>+}
  PP:{<IN><NP>}
  VP:{<VB.*><NP|PP|CLAUSE>+$}
  CLAUSE:{<NP><VP>}
"""
cp=nltk.RegexpParser(grammar)
sentence=[("Mary", "NN"), ("saw", "VBD"), ("the", "DT"), ("cat", "NN"),
("sit", "VB"), ("on", "IN"), ("the", "DT"), ("mat", "NN")]
print(cp.parse(sentence))

sentence1=[("John", "NNP"), ("thinks", "VBZ"), ("Mary", "NN"),
("saw", "VBD"), ("the", "DT"), ("cat", "NN"), ("sit", "VB"),
("on", "IN"), ("the", "DT"), ("mat", "NN")]
print(cp.parse(sentence1))

cp=nltk.RegexpParser(grammar,loop=2)
print(cp.parse(sentence1))

## Trees
tree1=nltk.Tree('NP',['Alice'])
print(tree1)

tree2=nltk.Tree('NP',['the','rabbit'])
print(tree2)

tree3=nltk.Tree('VP',['chased',tree2])
print(tree3)

tree4=nltk.Tree('S',[tree1,tree3])
print(tree4)

tree4.draw()

print(tree4[0])
tree4[1].label()
tree4[1].leaves()

## Tree Traversal

# 7.5 Named Entity Recognition
sent=nltk.corpus.treebank.tagged_sents()[22]
print(nltk.ne_chunk(sent,binary=True))
print(nltk.ne_chunk(sent))

# 7.6 Relation Extraction
import re
IN= re.compile(r'.*\bin\b(?!\b.+ing)')

# text='success the transition of'
# print(re.search(IN,text))

for doc in nltk.corpus.ieer.parsed_docs('NYT_19980315'):
    for rel in nltk.sem.extract_rels('ORG','LOC',doc,corpus='ieer',pattern=IN):
        print(nltk.sem.show_raw_rtuple(rel))
