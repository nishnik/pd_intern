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

# Remove punctuation and lower it
import re
import string
regex = re.compile('[%s]' % re.escape(string.punctuation))
for a in pubmed_fetch:
        pubmed_fetch[a]['abstract'] = regex.sub('', pubmed_fetch[a]['abstract'])
        pubmed_fetch[a]['abstract'] = pubmed_fetch[a]['abstract'].lower()

json.dump(pubmed_fetch, open('pubmed_fetch.json', 'w'))
json.dump(connection, open('connection_re.json', 'w'))