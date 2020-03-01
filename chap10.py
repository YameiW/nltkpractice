# 10.1 Natural language understanding
## Querying a Database
import nltk
nltk.data.show_cfg('grammars/book_grammars/sql0.fcfg')

from nltk import load_parser
cp=load_parser('grammars/book_grammars/sql0.fcfg')
query="What cities are located in China"
trees=next(cp.parse(query.split()))

answer=trees[0].label['sem'] # TypeError: 'method' object is not subscriptable
q=''.join(answer) #
print (q) #

from nltk.sem import chat80
rows=chat80.sql_query('corpora/city_database/city.db',q)
for r in rows:print(r[0])

## Natural language, Semantics, and logic

# 10.2 Propositional logic
nltk.boolean_ops()


# 10.3 First-order logic
dom=set(['b','o','c'])
g=nltk.Assignment(dom, [('x','o'),('y','c')])
g
v = """
    bertie => b
    olive => o
    cyril => c
    boy => {b}
    girl => {o}
    dog => {c}
    walk => {o, c}
    see => {(b, o), (c, b), (o, c)}
    """


print("test")


# 10.5 Discourse semantics