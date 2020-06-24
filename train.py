import torch
#from torch.autograd import Variable
import numpy as np
#import torch.nn as nn
import torch.functional as F
import torch.nn.functional as F
import optparse
import sys
from math import sqrt
from utils import *
from tqdm import tqdm
from fractions import Fraction
import random
from model import model
import time
import params


# Would implementing the sample on the skipgrams themselves have any effect?
# this seems very ineffective for certain texts
def subsample(sample, num_words):
    sampling_rate = .001
    # this is slow we could implement it once for each word
    p = sqrt((sample/num_words)/sampling_rate + 1) * sampling_rate/(sample/num_words)
    # this is choppy a larger seed might be a way to smooth it out
    rand = random.randint(1, 100)
    return((rand/p) >= 1)
    #return True


def create_distribution(vocab):
    
    dist = np.power(list(vocab.values()), 0.75)
    dist = dist / dist.sum()
    
    return torch.FloatTensor(dist)

# Corpus must be formatted such that it is a set with no punctuation marks and each sentence 
# is it's own seperate element
def train_model(corpus, epochs, save, use_gpu):
    
    tokenized_corpus = tokenize_corpus(corpus)
    vocabulary = build_vocabulary(tokenized_corpus)
    idx_pairs = create_skipgrams(tokenized_corpus)

    vocabulary_size = len(vocabulary)

    #TODO add hyper parameter configuration/create model class
    

    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')

    weight_dist = create_distribution(vocabulary)
 
    if params.batch_size > len(idx_pairs):
        params.batch_size = len(idx_pairs)

    wordModel = model(params.embedding_dims, vocabulary_size, weight_dist, params.batch_size, params.window_size, params.negative_samples, device).to(device)

    num_epochs = epochs

    optimizer = torch.optim.Adam([wordModel.i_vec.weight, wordModel.o_vec.weight], lr=learning_rate)

    print("Initializing training ....")
    print("Utilizing: " + str(device) + " ....")

    for epo in range(num_epochs):
        
        print(f"Epoch {epo} ...")
        loss_val = 0
        i_words = []
        o_words = []
        #start = time.time()
        for idx, words in tqdm(enumerate(idx_pairs, 1), total=len(idx_pairs)):
            i_words.append(words[0])
            o_words.append(words[1])

            if idx % params.batch_size == 0:
                #end = time.time()
                #print("Array appension: " + str(end - start))
                #start = time.time()
                loss = wordModel.forward(i_words, o_words)
                #end = time.time()
                #print("Forward Pass: " + str(end - start))
                optimizer.zero_grad()
                #start = time.time()
                loss.backward()
                #end = time.time()
                #print("BackProp: " + str(end - start))
                #start = time.time()
                optimizer.step()
                #end = time.time()
                #print("Optimizer Step: " + str(end - start))
                #start = time.time()
                loss_val += loss

                i_words = []
                o_words = []

        print(f'Loss at epoch {epo}: {loss_val/len(idx_pairs) * params.batch_size}')
    
    #TODO maybe add keyboard interrupts and incremental save
    save_model(wordModel, save)


if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("--epochs", action="store", help="epochs to run for", type='int')
    parser.add_option("--text", action="store", help="A .txt file or a directory with .txt files inside")
    parser.add_option("--save", action="store", default="polymnia", help="the model save file name under ./model")
    parser.add_option("--use-gpu", dest='gpu', action="store_true", help="use gpu accleration (requires cuda)")
    (args, _) = parser.parse_args(sys.argv)
    
    sentences = GetWords(args.text)

    train_model(sentences, args.epochs, args.save, args.gpu)
