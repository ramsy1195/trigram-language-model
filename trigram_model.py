import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2024 
Programming Homework 1 - Trigram Language Models
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
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    n_sequence = ['START'] + sequence + ['STOP']
    if n > 1:
        n_sequence = ['START'] * (n - 2) + n_sequence
    
    ngrams = []
    for i in range(len(n_sequence) - n + 1):
        ngram = tuple(n_sequence[i:i + n])
        ngrams.append(ngram)
        
    return ngram


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
        
        # Calculating total number of words
        self.total_word_count = sum(self.unigramcounts.values())


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        for sentence in corpus:
            unigrams = get_ngrams(sentence, 1)
            bigrams = get_ngrams(sentence, 2)
            trigrams = get_ngrams(sentence, 3)

            for unigram in unigrams:
                self.unigramcounts[unigram] += 1

            for bigram in bigrams:
                self.bigramcounts[bigram] += 1

            for trigram in trigrams:
                self.trigramcounts[trigram] += 1

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        bigram = (trigram[0], trigram[1])
        count_trigram = self.trigramcounts[trigram]
        count_bigram = self.bigramcounts[bigram]

        if count_bigram > 0:
            return count_trigram / count_bigram
        else:
            # Uniform distribution over vocabulary size |V|
            return 1 / len(self.lexicon)

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        first_unigram = (bigram[0],)
        count_bigram = self.bigramcounts[bigram]
        count_unigram = self.unigramcounts[first_unigram]

        if count_unigram > 0:
            return count_bigram / count_unigram
        else:
            return 0.0
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        if self.total_word_count > 0:
            return self.unigramcounts[unigram] / self.total_word_count
        else:
            return 0.0
        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        
        u = trigram[0]
        w = trigram[1]
        v = trigram[2]
    
        p_trigram = self.raw_trigram_probability(trigram)
        p_bigram = self.raw_bigram_probability((w, v))
        p_unigram = self.raw_unigram_probability((v,))
        
        smoothed_prob = (lambda1 * p_trigram) + (lambda2 * p_bigram) + (lambda3 * p_unigram)
        
        return smoothed_prob
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        log_prob = 0.0
        
        trigrams = get_ngrams(sentence, 3)
        
        for trigram in trigrams:
            prob = self.smoothed_trigram_probability(trigram)
            if prob > 0:
                log_prob += math.log2(prob)
            else:
                return float("-inf")  

        return log_prob

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the perplexity 
        """
        total_log_prob = 0.0
        total_tokens = 0
        total_sentences = 0

        for sentence in corpus:
            total_sentences += 1
            log_prob = self.sentence_logprob(sentence)
            total_log_prob += log_prob
            total_tokens += len(sentence)  

        if total_tokens > 0:
            l = total_log_prob / total_tokens
            perplexity = 2 ** (-l)
            return perplexity
        else:
            return float("inf") 


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)

    total = 0
    correct = 0       
 
    for f in os.listdir(testdir1):
        total += 1
        pp_high = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
        pp_low = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
        if pp_high < pp_low:
            correct += 1  
        else:
            pass
    
    for f in os.listdir(testdir2):
        total += 1
        pp_low = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
        pp_high = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
        if pp_low < pp_high:
            correct += 1  
        else:
            pass

    accuracy = correct / total if total > 0 else 0.0
    return accuracy

if __name__ == "__main__":

    # model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)
    
    # for testing purposes:
    # if len(sys.argv) < 5:
    #    print("Usage: python trigram_model.py <training_high> <training_low> <test_high> <test_low>")
    #    sys.exit(1)

    # training_high = sys.argv[1]
    # training_low = sys.argv[2]
    # test_high = sys.argv[3]
    # test_low = sys.argv[4]

    # model_high = TrigramModel(training_high)
    # model_low = TrigramModel(training_low)

    # dev_corpus = corpus_reader(test_high, model_high.lexicon)
    # pp_high = model_high.perplexity(dev_corpus)
    # print(f"Perplexity for high model on dev corpus: {pp_high}")

    # dev_corpus = corpus_reader(test_low, model_low.lexicon)
    # pp_low = model_low.perplexity(dev_corpus)
    # print(f"Perplexity for low model on dev corpus: {pp_low}")

    # accuracy = essay_scoring_experiment(training_high, training_low, test_high, test_low)
    # print(f"Accuracy of essay scoring experiment: {accuracy:.2f}")

