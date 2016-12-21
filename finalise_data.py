import json

connection = json.load(open('connection.json'))
pubmed_fetch = json.load(open('pubmed_fetch.json'))

for a in list(pubmed_fetch.keys()):
    if (not 'abstract' in pubmed_fetch[a].keys()):
        del pubmed_fetch[a]
        continue
    try:
        tmp = ""
        for b in pubmed_fetch[a]['abstract']:
            tmp += b + " "
        pubmed_fetch[a]['abstract'] = tmp[:-1]
    except:
        print (a)

import re
f = open('stopwords.txt', 'r')
stopwords = f.read()
list_stopwords = re.sub("[^\w]", " ",  stopwords).split()

# Remove punctuation and remove special chars and lower it
import string
regex = re.compile('[%s]' % re.escape(string.punctuation))
for a in pubmed_fetch:
    temp = regex.sub('', pubmed_fetch[a]['abstract'])
    temp = temp.lower()
    pubmed_fetch[a]['abstract'] = "".join([c for c in temp if c in string.ascii_lowercase or c in string.whitespace])

done = 0
for a in list(pubmed_fetch.keys()):
    print (done)
    done += 1
    words = re.sub("[^\w]", " ",  pubmed_fetch[a]['abstract']).split()
    pubmed_fetch[a]['abstract'] = ' '.join(word for word in words  if word not in list_stopwords)

to_vanish =  set(connection.keys()) - set(pubmed_fetch.keys())

for a in to_vanish:
    del connection[a]

for a in connection:
    for b in range(len(connection[a])):
        if (b < len(connection[a])):
            try:
                if connection[a][b] in to_vanish:
                    del connection[a][b]
            except:
                print (a, b)
                pass



json.dump(pubmed_fetch, open('pubmed_fetch.json', 'w'))
json.dump(connection, open('connection_re.json', 'w'))
