import sys
from collections import defaultdict
from collections import Counter
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Spring 2018
Homework 1 - Programming Component: Trigram Language Models
Daniel Bauer
"""
def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  

def get_ngrams(sequence, n):
    padded_list = []
    if n == 1:
        padded_list.append("START")
    else:
        t = 1
        while t<n:
            padded_list.append("START")
            t = t + 1
    k = 0
    while k < len(sequence):
            padded_list.append(sequence[k])
            k = k + 1
    padded_list.append("STOP")
    ngram = []
    ngrams = []
    i = 0 #iterates under n
    j = 0 #iterates over string
    while j < len(padded_list):
        while i < n:
            if (i+j) < len(padded_list):
                ngram.append(padded_list[j+i])
            i = i + 1
        if len(ngram) == n:
            ngram = tuple(ngram)
            ngrams.append(ngram)
        j = j + 1
        i = 0
        ngram = []
    return ngrams

class TrigramModel(object):
    
    def __init__(self, corpusfile):

        generator = corpus_reader(corpusfile)   
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)
        global unigramcounts

    def count_ngrams(self, corpus):

        global unigramcounts 
        global bigramcounts 
        global trigramcounts 
        unigramcounts = {} 
        bigramcounts = {} 
        trigramcounts = {} 
        unigrams = []
        bigrams = []
        trigrams = []
        global unigrams_total
        global bigrams_total
        global trigrams_total
        i = 0
        x =0 
        global words_total
        counts_starts_stops = 0

        for sentence in corpus:
            if x>-100: #line useful for testing e.g. by setting x<1000 
                x = x+1   
                padded_length = len(sentence) 
                pp = 0
                while i<=padded_length:
                    pp = pp + padded_length
                    trigrams.append(get_ngrams(sentence,3)[i])
                    bigrams.append(get_ngrams(sentence,2)[i])
                    if get_ngrams(sentence,1)[i] == ('START',):
                        counts_starts_stops +=1
                    if get_ngrams(sentence,1)[i] == ('STOP',):
                        counts_starts_stops +=1
                    unigrams.append(get_ngrams(sentence,1)[i])
                    i += 1
                i = 0
        unigrams_total = len(unigrams)
        bigrams_total = len(bigrams)
        trigrams_total = len(trigrams)
        words_total = unigrams_total +1 - counts_starts_stops
        unigrams_dict = Counter(unigrams)
        bigrams_dict = Counter(bigrams)
        trigrams_dict = Counter(trigrams)
        unigramcounts = unigrams_dict 
        bigramcounts = bigrams_dict
        trigramcounts = trigrams_dict
        return

    def raw_trigram_probability(self,trigram):  
        global trigramcounts
        global bigramcounts
        numerator   = trigramcounts[trigram]*1.0
        denominator = bigramcounts[(trigram[0], trigram[1])]*1.0
        if (trigram[0], trigram[1])== ('START', 'START'):
            raw_probability = (numerator) /((1.0*unigramcounts[('START',)]))
        elif numerator == 0.0:
            raw_probability = 0.0
        else:
            raw_probability = (numerator)/(denominator*1.0)
        return raw_probability

    def raw_bigram_probability(self, bigram):
        global unigramcounts
        global bigramcounts
        numerator   = bigramcounts[bigram]*1.0
        denominator = unigramcounts[(bigram[0],)]*1.0
        if numerator == 0.0:
            raw_probability = 0.0        
        else:
            raw_probability = (numerator)/((denominator*1.0))
        return raw_probability
    
    def raw_unigram_probability(self, unigram):
        global unigramcounts
        global words_total
        global unigrams_total
        unigram_count = unigramcounts[unigram]*1.0
        raw_probability = (unigram_count)/(unigrams_total*1.0)
        return raw_probability

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        lambda1 = 1.0/3.0
        lambda2 = 1.0/3.0
        lambda3 = 1.0/3.0
        smoothed_probability = (lambda1*self.raw_trigram_probability(trigram)+ 
                                lambda2*self.raw_bigram_probability((trigram[1], trigram[2]))
                                + lambda3*self.raw_unigram_probability((trigram[2],)))
        return smoothed_probability
        
    def sentence_logprob(self, sentence):
        logprob = 0.0 
        trigrams = get_ngrams(sentence,3) 
        for trigram in trigrams:
            log_arg = self.smoothed_trigram_probability(trigram)
            if log_arg == 0:
                logprob += 0
            else:
                logprob += math.log(log_arg,2)
        return logprob

    def perplexity(self, corpus):
        M = 0.0
        l = 0.0       
        generator = corpus 
        for sentence in generator:
            l += self.sentence_logprob(sentence)
            M += 1.0*len(sentence)
        l = l * (-1.0/M) 
        perplexity = 2.0 ** (l)
        return perplexity 


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
        #testing with model1
        model1 = TrigramModel(training_file1)
        pp_model1 = []
        pp_model2 = []
        total = 0.0
        correct = 0.0      
        acc = 0.0
        for f in os.listdir(testdir1):
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp_model1.append(pp1)
            total +=1.0
        for f in os.listdir(testdir2):
            pp2 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp_model1.append(pp2)
            total += 1.0
        #testing with model2
        model2 = TrigramModel(training_file2)    
        for f in os.listdir(testdir1):
            pp1 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            pp_model2.append(pp1)
        for f in os.listdir(testdir2):
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp_model2.append(pp2)
        #comparing stored perplexity values           
        j = 0
        i = 0
        while i < len(pp_model1):
            if j <= len(testdir1):
                if pp_model1[i] < pp_model2[i]:
                    correct +=1.0   
            else:
                if pp_model2[i] < pp_model1[i]:
                    correct +=1.0
            j += 1
            i += 1
        total = 1.0*len(pp_model2)
        acc = 1.0*(correct / total)
        return acc 

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 
    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    global unigramcounts
    global trigramcounts
    global bigramcounts
    model1 = TrigramModel("train_high.txt")
    dev_corpus = corpus_reader(sys.argv[1], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print pp, "pp_brown train"
    pp = model.perplexity(corpus_reader(sys.argv[2], model.lexicon))
    print pp, "pp_brown test"
    # Essay scoring experiment: 
    acc = essay_scoring_experiment("train_high.txt", "train_low.txt", "test_high", "test_low")
    print acc, "acc"



