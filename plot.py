from utils import *
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib.pyplot as plt
import optparse
import sys
import torch
import matplotlib
from threading import Thread


if __name__ == "__main__":

    parser = optparse.OptionParser()

    parser.add_option("--text", action="store", help="text body")
    parser.add_option("--model", action="store", help="model file")
    parser.add_option("--list", action="store", help="list of words to plot")
    parser.add_option("--cossim", action="store", help="show cosine similarity")

    (params, _) = parser.parse_args(sys.argv)

    matplotlib.rc("font", family="AppleGothic")

    corpus = GetWords(params.list)
    tokenized_corpus = tokenize_corpus(corpus)
    words = list(build_vocabulary(tokenized_corpus).keys())[1:]

    corpus = GetWords(params.text)
    tokenized_corpus = tokenize_corpus(corpus)
    vocabulary = build_vocabulary(tokenized_corpus)
    
    vocabulary_size = len(vocabulary)

    wordModel = load_model(params.model)
    

    if params.cossim is not None:
        for word, score in get_close_words(params.cossim, 30, words, wordModel):
            print((word, (1 - score)))
    else:
        vecs = []
        print("Getting word vectors ... ")
        for word in tqdm(words[1:]):
            vecs.append(get_word_vector(word, wordModel).squeeze())
        model = TSNE(n_components=2, perplexity=30, init='pca', method='exact', n_iter=5000)
        X = vecs
        # This is crashing with large sets because of a BLAS library issue (out of memory)
        print("Performing vector transform ... ")
        X = model.fit_transform(X)
        print("Creating plot ... ")
        plt.figure(figsize=(18, 18))
        for i in tqdm(range(len(X))):
            plt.text(X[i, 0], X[i, 1], words[i], bbox=dict(facecolor='blue', alpha=0.1))
        plt.xlim((np.min(X[:, 0]), np.max(X[:, 0])))
        plt.ylim((np.min(X[:, 1]), np.max(X[:, 1])))

        print("Saving plot ... ")
        
        plt.savefig('./output.png')
