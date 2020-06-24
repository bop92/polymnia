import re
import os
import itertools
import numpy as np
from tqdm import tqdm
from params import *
from scipy import spatial
import torch


#Recurse the subdirectory for text files
def GetFiles(path):

    textFiles = []

    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(".txt"):
                textFiles.append(root + "/" + name)
    return textFiles

'''
Homogonize the text files
'''
def GetWords(text):

    print("Parsing corpus ...")

    lines = []

    if os.path.isdir(text):

        files = GetFiles(text)

        print("Found " + str(len(files)) + " Text files ...")

        for text in files:

            text = open(text, "r")
            lines.extend(text.readlines())
            text.close()

    else:
        text = open(text, "r")
        lines = text.readlines()
        text.close()

    #TODO need to account for period and other punctuation
    # by this I mean that  we can generate better chains by considering
    # punctation marks as part of the structure

    out = []
    for line in lines:
        foo = []
        line = line.rstrip('\n')
        tokens = line.split(" ")
        tokens = map(lambda x : x.split("-"), tokens)
        tokens = list(itertools.chain.from_iterable(tokens))
        tokens = list(tokens)
        for token in tokens:
            token = token.lower()
            token = re.sub('[^a-zA-Z]+', '', token)
            foo.append(token)
        line = " ".join(foo)
        out.append(line.rstrip("\n"))
    return out


def tokenize_corpus(corpus):
    tokens = [x.split() for x in corpus]
    return tokens


def build_vocabulary(tokenized_corpus):

    vocabulary = {}
    vocabulary[unk] = 1
    print("Creating word index ... ")

    for sentence in tqdm(tokenized_corpus):
        for token in sentence:
            if not token in vocabulary:
                vocabulary[token] = 1
            else:
                vocabulary[token] += 1
    global word2idx
    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}
    global idx2word
    idx2word = {idx: w for (idx, w) in enumerate(vocabulary)}

    global vocabulary_size
    vocabulary_size = len(vocabulary)

    return vocabulary

def create_skipgrams(tokenized_corpus):

    # A 2 skip might not be the best way to model this
    # there might be some fancy way to do this with a probablisitic assesment
    # (other than subsampling)

    # Chaining word pairs might also be a solution but it
    # will grow the vocabulary
    # "the" "dog" -> "the dog"

    idx_set = []
    idx_pairs = []
    print("Creating skip grams ... ")

    for sentence in tqdm(tokenized_corpus):
        for idx, token in enumerate(sentence):
            idx_set = []
            for i in range(window_size, 0, -1):
                if (idx-i) >= 0:
                    add = sentence[idx-i]
                else:
                    add = unk
                idx_set.append(word2idx[add])
            for i in range(1, window_size + 1):
                if (idx+i) < len(sentence):
                    add = sentence[idx+i]
                else:
                    add = unk
                idx_set.append(word2idx[add])
            idx_set = np.array(idx_set)
            idx_pairs.append((word2idx[token], idx_set))

    return idx_pairs


def get_word_vector(word, model):

    try:
        x = np.array([word2idx[word]])
    except KeyError:
        print(f"No key correlating to word {word}")
        print("Exiting ... ")
        exit(1)
    z1 = model.i_vec.to("cpu")

    z1 = z1(torch.LongTensor(x))

    return(z1.cpu().squeeze().detach().numpy())


def get_close_words(word, num, vocabulary, model):
    
    vecs = []
    for mword in vocabulary:
        w = get_word_vector(word, model)
        m = get_word_vector(mword, model)
        vecs.append(spatial.distance.cosine(w,m))
    
    words = sorted(list(enumerate(vecs)), key=lambda x: x[1])
    
    out = []
    #for i in range(len(words) - 1, len(words) - num - 1, -1):
    for i in range(0, num):
        tup = words[i]
        out.append((idx2word[tup[0]], tup[1]))
    
    return out


def load_model(name):
    print("Loading model ... ")
    try:
        model = torch.load(name)
        model.eval() # need to add training flag too
    except FileNotFoundError:
        print("Model file not found ... exiting")
        exit(1)

    return model
    
def save_model(wordModel, save):    
    if not os.path.exists("./model"):
        os.mkdir("./model")
    torch.save(wordModel, "./model/" + save + ".model")
