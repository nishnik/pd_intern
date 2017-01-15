import json
import gensim
from gensim import corpora
import numpy as np
connection = json.load(open('connection_re.json'))
pubmed_fetch = json.load(open('pubmed_fetch.json'))
all_keys = list(connection.keys())
train_till = int(0.8 * len(all_keys))

documents = []

for a in all_keys:
	documents.append(pubmed_fetch[a]['abstract'])

documents = [[word for word in document.lower().split()] for document in documents]
dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(text) for text in texts]
lda = gensim.models.ldamodel.LdaModel(corpus=corpus[:train_till], id2word=dictionary, num_topics=128, update_every=1, chunksize=10000, passes=1)

dict_repr = {}
count = 0
for i in range(len(all_keys)):
    count += 1
    print (count)
    tmp = lda[corpus[i]]
    dict_repr[all_keys[i]] = [0.0]*128
    for a in tmp:
    	dict_repr[all_keys[i]][a[0]] = a[1]


count = 0
dict_top = {}
dict_top[100] = {}
for a in dict_repr:
    count += 1
    print (count)
    vec1 = dict_repr[a]
    top_100 = {}
    for b in dict_repr:
        if not b == a:
            val = np.dot(vec1, dict_repr[b])
            if len(top_100) < 100:
                top_100[b] = val
            else:
                m = min(top_100, key=top_100.get)
                if (val > top_100[m]):
                    del top_100[m]
                    top_100[b] = val
    dict_top[100][a] = top_100



count = 0

for a in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
    dict_top[a]= {}

for a in dict_top[100]:
    count += 1
    print (count)
    for num in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
        tmp = sorted(dict_top[100][a].iteritems(), key = operator.itemgetter(1), reverse = True)
        dict_top[num][a] = dict(tmp[:num])


for ORDER in [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    p_f, p_t, t_f = 0, 0, 0
    for a in all_keys:
        count_real = 0
        for aa in dict_top[ORDER][a]:
             if aa in connection[a]:
                count_real += 1
        p_f += count_real
        p_t += min(len(connection[a]), ORDER)
        t_f += ORDER
    recall = float(p_f)/p_t
    precision = float(p_f)/t_f
    f1 = 2*precision*recall/(precision + recall)
    print ("ORDER: ", ORDER, recall, precision, f1)
