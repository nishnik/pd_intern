import numpy as np
import json
from keras import backend
from keras.layers import Input, merge
from keras.layers.core import Dense, Lambda, Reshape
from keras.layers.convolutional import Convolution1D
from keras.models import Model
import string
import re
import random
import operator
from itertools import izip
## Here connection is a dict which gives positively related documents id
## {'12' : ['13', '14'], '13' : ['14', '15']}
##
## Pubmed_fetch contains the abstract and tite for a given id i.e
## {'12' : {'title' : 'title of 12', 'abstract' : 'abstract of 12'}}
connection = json.load(open('connection_re.json'))
pubmed_fetch = json.load(open('pubmed_fetch_gens.json'))
all_keys = list(pubmed_fetch.keys())

WINDOW_SIZE = 3 # this is to decide window for words in a sentence (sliding window) See section 3.2
TOTAL_LETTER_GRAMS = 200 # Determined from data. See section 3.2. (26(abc..)+1(#)+1(?))^3
WORD_DEPTH = WINDOW_SIZE * TOTAL_LETTER_GRAMS # See equation (1).
K = 300 # Dimensionality of the max-pooling layer. See section 3.4.
L = 128 # Dimensionality of latent semantic space. See section 3.5.
J = 4 # Number of random unclicked documents serving as negative examples for a query. See section 4.
FILTER_LENGTH = 2 # We only consider one time step for convolutions.

def load_emb(types_file, emb_file):
    emb = {}
    with open(types_file, 'r') as types_f, open(emb_file, 'r') as emb_f:
            for line1, line2 in izip(types_f, emb_f):
                line2 = line2.strip()
                emb[line1.strip('\r\n').decode('UTF-8', 'ignore')] = np.array(map(float, line2.split(' ')), dtype=np.float32)
    return emb

count = 0
corpus = set()
for a in list(pubmed_fetch.keys()):
    count += 1
    print count
    for b in pubmed_fetch[a]['abstract'].lower().split():
        corpus.add(b)


emb = load_emb("types.txt", "vectors.txt")

for a in corpus:
    if not a in emb:
        print (a)


emb_keys = emb.keys()
for a in emb_keys:
    if not a in corpus:
        del emb[a]





## gives you a numpy.ndarray that is the vector for given sentence
def get_vector(sentence):
    ## get words in the sentence
    words = sentence.split()
    output_vec = [] # size will len(words) - 2, and each element will have size of WORD_DEPTH
    sliding_window = [] # size will be WINDOW_SIZE, and each element will have size of TOTAL_LETTER_GRAMS
    for ind in range(len(words)):
        word = words[ind]
        # if (len(word) >= 3):
        word_vec = emb[word]
        if (ind < WINDOW_SIZE-1):
            sliding_window.append(word_vec)
        else:
            sliding_window.append(word_vec)
            temp = sliding_window[0]
            for s in sliding_window[1:]:
                temp = np.concatenate((temp, s))
            output_vec.append(temp)
            del temp
            del sliding_window[0]
    del sliding_window
    return np.array(output_vec)


train_till = int(0.8 * len(all_keys))
## Get random negative doc ids
def get_negatives(pmid):
    output_ids = []
    while (len(output_ids) < J):
        ind = random.randrange(0, train_till)
        if not all_keys[ind] in connection[pmid]:
            output_ids.append(all_keys[ind])
    return output_ids


def R(vects):
    """
    Calculates the cosine similarity of two vectors.
    :param vects: a list of two vectors.
    :return: the cosine similarity of two vectors.
    """
    (x, y) = vects
    return backend.dot(x, backend.transpose(y)) / (x.norm(2) * y.norm(2)) # See equation (4)


# Input tensors holding the query, positive (clicked) document, and negative (unclicked) documents.
# The first dimension is None because the queries and documents can vary in length.
query = Input(shape = (None, WORD_DEPTH))
pos_doc = Input(shape = (None, WORD_DEPTH))
neg_docs = [Input(shape = (None, WORD_DEPTH)) for j in range(J)]

# In this step, we transform each word vector with WORD_DEPTH dimensions into its
# convolved representation with K dimensions. K is the number of kernels/filters
# being used in the operation. Essentially, the operation is taking the dot product
# of a single weight matrix (W_c) with each of the word vectors (l_t) from the
# query matrix (l_Q), adding a bias vector (b_c), and then applying the tanh function.
# That is, h_Q = tanh(W_c * l_Q + b_c). With that being said, that's not actually
# how the operation is being calculated here. To tie the weights of the weight
# matrix (W_c) together, we have to use a one-dimensional convolutional layer. 
# Further, we have to transpose our query matrix (l_Q) so that time is the first
# dimension rather than the second (as described in the paper). That is, l_Q[0, :]
# represents our first word vector rather than l_Q[:, 0]. We can think of the weight
# matrix (W_c) as being similarly transposed such that each kernel is a column
# of W_c. Therefore, h_Q = tanh(l_Q * W_c + b_c) with l_Q, W_c, and b_c being
# the transposes of the matrices described in the paper.

# Next, we apply a max-pooling layer to the convolved query matrix. Keras provides
# its own max-pooling layers, but they cannot handle variable length input (as
# far as I can tell). As a result, I define my own max-pooling layer here. In the
# paper, the operation selects the maximum value for each row of h_Q, but, because
# we're using the transpose, we're selecting the maximum value for each column.

# In this step, we generate the semantic vector represenation of the query. This
# is a standard neural network dense layer, i.e., y = tanh(W_s * v + b_s).

doc_conv = Convolution1D(K, FILTER_LENGTH, border_mode = "same", input_shape = (None, WORD_DEPTH), activation = "tanh")
doc_max = Lambda(lambda x: x.max(axis = 1), output_shape = (K, ))
doc_sem = Dense(L, activation = "tanh", input_dim = K)


query_conv = doc_conv(query) # See equation (2).
query_max = doc_max(query_conv) # See section 3.4.
query_sem = doc_sem(query_max) # See section 3.5.

pos_doc_conv = doc_conv(pos_doc)
pos_doc_max = doc_max(pos_doc_conv)
pos_doc_sem = doc_sem(pos_doc_max)

neg_doc_convs = [doc_conv(neg_doc) for neg_doc in neg_docs]
neg_doc_maxes = [doc_max(neg_doc_conv) for neg_doc_conv in neg_doc_convs]
neg_doc_sems = [doc_sem(neg_doc_max) for neg_doc_max in neg_doc_maxes]


# This layer calculates the cosine similarity between the semantic representations of
# a query and a document.
R_layer = Lambda(R, output_shape = (1, )) # See equation (4).

# Returns the final 128 Dimensional vector
def return_repr(vects):
    return vects[0]

repr = Lambda(return_repr, output_shape = (1, 128))
repr_vect = repr([query_sem])

R_Q_D_p = R_layer([query_sem, pos_doc_sem]) # See equation (4).
R_Q_D_ns = [R_layer([query_sem, neg_doc_sem]) for neg_doc_sem in neg_doc_sems] # See equation (4).

concat_Rs = merge([R_Q_D_p] + R_Q_D_ns, mode = "concat")
## J = negative docs number, 1 = pos docs number
concat_Rs = Reshape((J + 1, 1))(concat_Rs)

# In this step, we multiply each R(Q, D) value by gamma. In the paper, gamma is
# described as a smoothing factor for the softmax function, and it's set empirically
# on a held-out data set. We're going to learn gamma's value by pretending it's
# a single, 1 x 1 kernel.
weight = np.array([1]).reshape(1, 1, 1, 1)
with_gamma = Convolution1D(1, 1, border_mode = "same", input_shape = (J + 1, 1), activation = "linear", bias = False, weights = [weight])(concat_Rs) # See equation (5).

# Next, we exponentiate each of the gamma x R(Q, D) values.
exponentiated = Lambda(lambda x: backend.exp(x), output_shape = (J + 1, ))(with_gamma) # See equation (5).
exponentiated = Reshape((J + 1, ))(exponentiated)

# Finally, we use the softmax function to calculate the P(D+|Q).
prob = Lambda(lambda x: x[0][0] / backend.sum(x[0]), output_shape = (1, ))(exponentiated) # See equation (5).

# We now have everything we need to define our model.
model = Model(input = [query, pos_doc] + neg_docs, output = prob)
model.compile(optimizer = "adadelta", loss = "binary_crossentropy")

# Because we're using the "binary_crossentropy" loss function, we can pretend that
# we're dealing with a binary classification problem and that every sample is a
# member of the "1" class.
y = np.ones(1)

for ind in range(train_till):
    try:
        print (ind+1, "/", train_till)
        i = all_keys[ind]
        l_Qs = get_vector(pubmed_fetch[i]['abstract'].lower())
        ## For making it compatible with model, we reshape it
        l_Qs = l_Qs.reshape(1, l_Qs.shape[0], l_Qs.shape[1])
        for jind in range(min(len(connection[i]), 6)):
            j = connection[i][jind]
            pos_l_Ds = get_vector(pubmed_fetch[j]['abstract'].lower())
            pos_l_Ds = pos_l_Ds.reshape(1, pos_l_Ds.shape[0], pos_l_Ds.shape[1])
            neg_l_Ds = []
            for a in get_negatives(i):
                temp = get_vector(pubmed_fetch[a]['abstract'].lower())
                neg_l_Ds.append(temp.reshape(1, temp.shape[0], temp.shape[1]))   
            history = model.fit([l_Qs, pos_l_Ds] + neg_l_Ds, y, nb_epoch = 1, verbose = 1)
        ## save evry 1000 iterations
        if (ind % 1000 == 0):
            model.save_weights("model_gens.h5")
            print ("saved")
    except Exception as e:
        print (ind, all_keys[ind], e)
        continue


### TESTING ###


get_repr = backend.function([query], repr_vect)
# 128-D representation of the abstracts
dict_repr = {}

count = 0
# model.load_weights("model_final.h5")
for a in pubmed_fetch.keys():
    count += 1
    print (count)
    l_Qs = get_vector(pubmed_fetch[a]['abstract'].lower())
    l_Qs = l_Qs.reshape(1, l_Qs.shape[0], l_Qs.shape[1])
    dict_repr[a] = get_repr([l_Qs])

count = 0
# Top similar documents
dict_top = {}
# Top 100
dict_top[100] = {}
for a in all_keys[train_till:]:
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
    for a in all_keys[train_till:]:
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
