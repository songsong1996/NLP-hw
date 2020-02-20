import sys
from collections import defaultdict
import math
import random
import os
import os.path
import numpy as np
"""
COMS W4705 - Natural Language Processing - Spring 2020
Homework 1 - Programming Component: Trigram Language Models
Yassine Benajiba
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
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    seq_len=len(sequence)
    ngrams=[]
    if n==1:
        ngrams.append(('START',))
    if seq_len+2<n:
        return False
    else:
        for i in range(seq_len):
            if i<n-1:
                ngrams.append(('START',)*(n-i-1)+tuple(sequence[:i+1]) )   
            else:
                ngrams.append( tuple(sequence[i-n+1:i+1]))
        tp=ngrams[-1]
        ngrams.append( tuple(tp[1:])+('STOP',) )
    return ngrams


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {} 

        self.unigramn=0
        self.bigramn=0
        self.trigramn=0

        ##Your code here
        def get_count(ngram_lst, n,gram_dict):
            for ngram in ngram_lst:
                if n==1:
                    self.unigramn+=1
                elif n==2:
                    self.bigramn+=1
                else:
                    self.trigramn+=1
                if ngram not in gram_dict:
                    gram_dict[ngram]=1
                else:
                    gram_dict[ngram]+=1

        for sentense in corpus:
                ngram_lst=get_ngrams(sentense, 1)
                get_count( ngram_lst,1,self.unigramcounts)
                ngram_lst=get_ngrams(sentense, 2)
                get_count( ngram_lst,2,self.bigramcounts)
                ngram_lst=get_ngrams(sentense, 3)
                get_count( ngram_lst,3,self.trigramcounts)
        return


    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """

        # return self.trigramcounts[trigram]/self.trigramn
        if trigram not in self.trigramcounts:
                return 0
        if tuple(trigram[:-1])==('START','START'):
            return self.trigramcounts[trigram]/self.unigramcounts[('START',)]
        else:
            return self.trigramcounts[trigram]/self.bigramcounts[tuple(trigram[:-1])]

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        # return self.bigramcounts[bigram]/self.bigramn
        if bigram not in self.bigramcounts:
            return 0
        else:
            return self.bigramcounts[bigram]/self.unigramcounts[tuple(bigram[:-1])]
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        return self.unigramcounts[unigram]/self.unigramn

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        def FindNext(trigram,mode=0):
            tri_lst={}
            if mode==1:
                for tri in self.trigramcounts:
                    if tri[0:2]==tuple(['START']*2):
                        tri_lst[tri]=self.raw_trigram_probability(tri) 

                # return tri_lst[ np.random.randint(0,len(tri_lst),1)[0,0] ]
            else:
                for tri in self.trigramcounts:
                    if tri[0:2]==trigram[1:3]:
                        tri_lst[tri]=self.raw_trigram_probability(tri) 
            # prob_lst=
            random_tri_index= np.argmax(np.random.multinomial(1, list(tri_lst.values()),1))
            next_tri=list(tri_lst.keys())[random_tri_index]
            return next_tri
                # return tri_lst[ np.random.randint(0,len(tri_lst),1)[0,0] ]
        
        sentence=[]
        tri=FindNext([''],1)
        while 1:
            sentence.append(tri[2])
            if tri[-1]=='STOP':
                break
            else:
                tri=FindNext(tri)
            if len(sentence)>t:
                break
        return sentence


    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        prob=lambda1*self.raw_trigram_probability(trigram)+lambda2*self.raw_bigram_probability(tuple(trigram[1:]))+lambda3*self.raw_unigram_probability(tuple([trigram[-1]]))
        return prob
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigram_lst=get_ngrams(sentence,3)
        sen_prob=0
        for trigram in trigram_lst:
            sen_prob+=math.log2(self.smoothed_trigram_probability(trigram))
        return sen_prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        
        l=0
        m=0
        for sentence in corpus:
            l+=self.sentence_logprob(sentence)
            uni=get_ngrams(sentence,1)
            m+=len(uni)+1
        l/=m

        return 2**(-l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            total+=1
            p_h = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            p_l=model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if p_h<p_l:
                correct+=1
            # .. 

        for f in os.listdir(testdir2):
            total+=1
            p_h = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            p_l=model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            if p_l<p_h:
                correct+=1 
        
        return correct/total

if __name__ == "__main__":
    model=TrigramModel('brown_train.txt')
    # print(1)
    # model = TrigramModel(sys.argv[1]) 

    model_corpus=corpus_reader('brown_train.txt',model.lexicon)
    print("train_set_perplexity=",model.perplexity(model_corpus))
  
    # print(1)
    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 
    
    ## Testing perplexity: 
    dev_corpus = corpus_reader('brown_test.txt', model.lexicon)
    pp = model.perplexity(dev_corpus)
    print("test_set_perplexity=",pp)
    print("essay_score_accuracy=",essay_scoring_experiment('ets_toefl_data/train_high.txt', 'ets_toefl_data/train_low.txt', 'ets_toefl_data/test_high', 'ets_toefl_data/test_low'))
   
    #print random sentences
    print("randomly generated sentences: ")
    print(model.generate_sentence())
    print(model.generate_sentence())
    print(model.generate_sentence())

    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt', 'test_high', 'test_low')
    # print(acc)

