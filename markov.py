#!/usr/bin/env python3

import sys
import optparse
import random
from utils import *

#In general this model could be improved through some kind of ordering
#mechanism like that found in a Hidden Markov Model

def freq2prob(markovfreq):
    
    for frag in markovfreq:
        fragSum = sum(markovfreq[frag].values())

        for subfrag in markovfreq[frag]:

            markovfreq[frag][subfrag] = float(markovfreq[frag][subfrag] / fragSum)

    return markovfreq



def MarkovModel(sentence, order, MyModel):

    sentence = sentence.split(" ")
    
    for i in range(0, len(sentence) - order):
        curr_words = " ".join(sentence[i:i + order])
        next_word = sentence[i + order]

        if curr_words not in MyModel:
            MyModel[curr_words] = {}

        if next_word not in MyModel[curr_words]:
            MyModel[curr_words][next_word] = 1

        else:
            MyModel[curr_words][next_word] += 1

    return MyModel



def ExpandMM(myList, order):

    ExpandedModel = {}

    for i in range(1, order + 1):

        curr_order = i

        for j in range(len(myList)):

            ExpandedModel = MarkovModel(myList[j], curr_order, ExpandedModel)
        
    return ExpandedModel


def weighted_choice(picks):

    tot = sum(weight for pick, weight in picks)

    rand = random.uniform(0, tot)

    upto = 0

    for pick, weight in picks:

        if upto + weight > rand:
            return pick

        upto += weight



def generateWords(order, length, text):
    
    myList = GetWords(text)

    mModel = freq2prob(ExpandMM(myList, order))
    keys = list(mModel.keys())

    starts = [keys[x]
                for x in range(len(keys))
                if len(keys[x].split(" ")) == 1]

    phrase = ''
    phrases = []
    i = 1

    while length > 0:
        
        if len(phrase) == 0:
            pick = random.choice(starts)
            phrase += pick
        
        elif phrase in keys:
            pick = weighted_choice(mModel[phrase].items())
            phrase = phrase + " " + pick

        else:
            phrase = " ".join(phrase.split(" ")[i:order])
            i += 1

        if pick != ''.join(phrases[-1:]):
            phrases.append(pick)
            length -= 1

    print(" ".join(phrases))


if __name__ == '__main__':
    parser = optparse.OptionParser()
    parser.add_option("--length", action="store", help="text length", type='int')
    parser.add_option("--order", action="store", help="markov order", type="int")
    parser.add_option("--text", action="store", help="text body")

    (params, _) = parser.parse_args(sys.argv)

    generateWords(params.order, params.length, params.text)
