# 8.1 Some grammatical dilemmas
## linguistic data and unlimited possibilities
## Ubiquitous ambiguity
import nltk
groucho_grammar=nltk.CFG.fromstring(""" # CFG:context-free grammar
    S -> NP VP
    PP -> P NP
    NP -> Det N | Det N PP | 'I'
    VP -> V NP | VP PP
    Det -> 'an' | 'my'
    N -> 'elephant' | 'pajamas'
    V -> 'shot'
    P -> 'in'
    """)

sent=['I','shot','an','elephant','in','my','pajamas']
parser=nltk.ChartParser(groucho_grammar)
trees=parser.parse(sent)
for tree in trees:
    print(tree)

# 8.2 What's the use of syntax
## Beyond n-grams

# 8.3 Context-free grammar
## a simple grammar
grammar1=nltk.CFG.fromstring("""
    S -> NP VP
    VP -> V NP | V NP PP
    PP -> P NP
    V -> "saw" | "ate" | "walked"
    NP -> "John" | "Mary" | "Bob" | Det N | Det N PP
    Det -> "a" | "an" | "the" | "my"
    N -> "man" | "dog" | "cat" | "telescope" | "park"
    P -> "in" | "on" | "by" | "with"
""")
sent='Mary saw Bob'.split()
rd_parser=nltk.RecursiveDescentParser(grammar1)

for tree in rd_parser.parse(sent):
    print(tree)

sent1='the dog saw a man in the park'.split()
for tree in rd_parser.parse(sent1):
    print(tree.draw())


## writing your own grammars
## recursion in syntactic structure
grammar2=nltk.CFG.fromstring("""
    S -> NP VP
    NP -> Det Nom | PropN
    Nom -> Adj Nom | N
    VP -> V Adj | V NP | V S | V NP PP
    PP -> P NP
    PropN -> 'Buster' | 'Chatterer' | 'Joe'
    Det -> 'the' | 'a'
    N -> 'bear' | 'squirrel' | 'tree' | 'fish' | 'log'
    Adj -> 'angry' | 'frightened' | 'little' | 'tall'
    V -> 'chased' | 'saw' | 'said' | 'thought' | 'was' | 'put'
    P -> 'on'
    """)


# 8.4 Parsing with Context-free grammar
## Recursive Descent Parsing
rd_parser=nltk.RecursiveDescentParser(grammar1)
sent='Mary saw a dog'.split()
for t in rd_parser.parse(sent):
    print(t)

nltk.app.rdparser()

## Shift-Reduce Parsing
sr_parse=nltk.ShiftReduceParser(grammar1)
sent='Mary saw a dog'.split()
for t in sr_parse.parse(sent):
    print(t)

nltk.app.srparser()

## The left-Corner Parser
## Well formed substring tables
def init_wfst(tokens,grammar):
    numtokens=len(tokens)
    wfst=[[None for i in range (numtokens+1)] for j in range(numtokens+1)]
    for i in range (numtokens):
        productions=grammar.productions(rhs=tokens[i])
        wfst[i][i+1]=productions[0].lhs()
    return wfst

def complete_wfst(wfst,tokens,grammar,trace=False):
    index=dict((p.rhs(),p.ljs()) for p in grammar.productions())
    numtokens=len(tokens)
    for span in range(2,numtokens+1):
        for start in range(numtokens+1-span):
            end=start+span
            for mid in range(start+1,end):
                nt1,nt2=wfst[start][mid],wfst[mid][end]
                if nt1 and nt2 and (nt1,nt2) in index:
                    wfst[start][end]=index[(nt1),(nt2)]
                if trace:
                    print('[%s] %3s [%s] %3s [%s] ==> [%s]%3s [%s]'%\
                        (start,nt1,mid,nt2,end,start,index[(nt1,nt2)],end))
    return wfst

def display(wfst,tokens):
    print('\nWFST'+''.join([('%-4d'%i) for i in range (1,len(wfst))]))
    for i in range(len(wfst)-1):
        print("%d "%i),
        for j in range(1,len(wfst)):
            print ("%-4s "%(wfst[i][j] or '.')),
        print

tokens='I shot an elephant in my pajamas'.split()
wfst0=init_wfst(tokens,groucho_grammar)
display(wfst0,tokens)  # not a matrix?
wfst1=complete_wfst(wfst0,tokens,groucho_grammar)
display(wfst1,tokens)

# 8.5 Dependencies and Dependency Grammar

groucho_dep_grammar=nltk.DependencyGrammar.fromstring("""
    'shot' -> 'I' | 'elephant' | 'in'
    'elephant' -> 'an' | 'in'
    'in' -> 'pajamas'
    'pajamas' -> 'my'
    """)
 print(groucho_dep_grammar)

pdp=nltk.ProjectiveDependencyParser(groucho_dep_grammar)
sent='I shot an elephant in my pajamas'.split()
trees=pdp.parse(sent)
for tree in trees:
    print(tree)

## Valency and the lexicon
## Scaling up


# 8.6 Grammar Development
## treebanks and grammars
from nltk.corpus import treebank
t=treebank.parsed_sents('wsj_0001.mrg')[0]
print(t)

def filter(tree):
    child_nodes=[child.label() for child in tree if isinstance(child,nltk.Tree)]
    return (tree.label()=='VP') and ('S' in child_nodes)

[subtree for tree in treebank.parsed_sents() for subtree in tree.subtrees(filter)]

entries=nltk.corpus.ppattach.attachments('training')
table=nltk.defaultdict(lambda:nltk.defaultdict(set))

for entry in entries:
    key=entry.noun1+'-'+entry.prep+'-'+entry.noun2
    table[key][entry.attachment].add(entry.verb)

for key in sorted(table):
    if len(table[key])>1:
        print(key,'N:',sorted(table[key]['N']),'V:',sorted(table[key]['V']))

nltk.corpus.sinica_treebank.parsed_sents()[3450].draw()

## pernicious ambiguity
grammar=nltk.CFG.fromstring("""
    S -> NP V NP
    NP -> NP Sbar
    Sbar -> NP V
    NP ->'fish'
    V -> 'fish'
    """)

tokens=['fish']*5
cp=nltk.ChartParser(grammar)
for tree in cp.parse(tokens):
    print(tree)

## weighted grammar
def give(t):
    return t.label()=='VP' and len(t)>2 and t[1].label()=='NP'\
        and (t[2].label()=='PP-DTV' or t[2].label()=='NP')\
        and ('give' in t[0].leaves() or 'gave' in t[0].leaves())

def sent(t):
    return ''.join(token for token in t.leaves() if token[0] not in '*-0')

def print_node(t,width):
    output='%s %s: %s /%s:%s'\
        (sent(t[0]),t[1].label(),sent(t[1]),t[2].label(),sent(t[2]))
    if len(output)>width:
        output=output[:width]+"..."
    print(output)

for tree in nltk.corpus.treebank.parsed_sents():
    for t in tree.subtrees(give):
        print_node(t,72)

grammar=nltk.PCFG.fromstring("""
S -> NP VP          [1.0]
VP -> TV NP         [0.4]
VP -> IV            [0.3]
VP -> DatV NP NP    [0.3]
TV -> 'saw'         [1.0]
IV -> 'ate'         [1.0]
DatV -> 'gave'      [1.0]
NP -> 'telescopes'  [0.8]
NP -> 'Jack'        [0.2]
    """)

print(grammar)
viterbi_parser=nltk.ViterbiParser(grammar)
result=viterbi_parser.parse(['Jack','saw','telescopes'])
for i in result:
    print(i)




