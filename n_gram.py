"""
n-gram language model for Assignment 2: Starter code.
"""

import os
import sys
import argparse
from typing import Dict, List, Any
from tqdm import tqdm
from collections import Counter
import numpy as np

from load_data import load_data
from perplexity import evaluate_perplexity
from wer import rerank_sentences_for_wer, compute_wer

def get_args():
    """
    You may freely add new command line arguments to this function.
    """
    parser = argparse.ArgumentParser(description='n-gram model')
    parser.add_argument('-t', '--tokenization_level', type=str, default='character',
                        help="At what level to tokenize the input data")
    parser.add_argument('-n', '--n', type=int, default=1,
                        help="The value of n to use for the n-gram model")

    parser.add_argument('-e', '--experiment_name', type=str, default='testing',
                        help="What should we name our experiment?")
    parser.add_argument('-s', '--num_samples', type=int, default=10,
                        help="How many samples should we get from our model??")
    parser.add_argument('-x', '--max_steps', type=int, default=40,
                        help="What should the maximum output length of our samples be?")
    # stuff we added:
    parser.add_argument('-sm', '--smoothing', type=str, default='none',
                        help="which smoothing function to use for probability stuff")
                        # ^ choose from: "none", "laplace", "addk", "linearinterpolation", "label_smooth"
    parser.add_argument('-m', '--mode', type=str, default='default',
                        help="choose either 'default' or 'full_experiment' mode")

    args = parser.parse_args()
    return args

class NGramLM():
    """
    N-gram language model
    """

    def __init__(self, n:int, smoothing:str="none"):
        """
        Initializes the n-gram model. You may add keyword arguments to this function
        to modify the behavior of the n-gram model. The default behavior for unit tests should
        be that of an n-gram model without any label smoothing.

        Important for unit tests: If you add <bos> or <eos> tokens to model inputs, this should 
        be done in data processing, outside of the NGramLM class. 

        Inputs:
            n (int): The value of n to use in the n-gram model
        """
        print(f"~ initializing n-gram model with n={n} and sm={smoothing}")
        self.n = n
        self.grams = Counter()
        self.unigrams = Counter()
        
        self.smoothing = smoothing
        self.smoothing_factor = 0.1
        self.addk_k = 0.5
        self.lambdas = [1/n for _ in range(n)]  # for linear interpolation smoothing # @QQ need to calculate these using held-out set or something idk
        
        self.leftpdx = "*"  # left padding token
        self.unk = 'unk'  # token for unknown words

    def log_probability(self, model_input: List[Any], base=np.e):
        """
        Returns the log-probability of the provided model input.

        Inputs:
            model_input (List[Any]): The list of tokens associated with the input text.
            base (float): The base with which to compute the log-probability
        """        
        n = self.n
        count = self.grams.total()
        log_prob_sum = 0.0

        # insert as many left-padding tokens as needed
        x = [*model_input]
        for i in range(n-1):
            x.insert(0, self.leftpdx)

        # for each token in the input sequence:
        for i in range(len(x)-n+1):
            ngram = tuple([str(w) for w in x[i:i+n]])
            ngram_count = self.grams[ngram]

            if n >= 2:
                nm1gram = tuple([str(w) for w in x[i:i+n-1]])
                count = self.grams[nm1gram]

            # do probability stuff + smoothing
            
            if self.smoothing == "laplace":
                # P(xi | xi-1) = c(xi-1,xi) + 1 / c(xi-1) + |V|
                V = len(self.unigrams.keys())
                prob = (ngram_count + 1) / (count + V)
            
            elif self.smoothing == "addk":
                # P(xi | xi-1) = c(xi-1,xi) + k / c(xi-1) + k*|V|
                V = len(self.unigrams.keys())
                prob = (ngram_count + self.addk_k) / (count + (self.addk_k * V))
            
            elif self.smoothing == "linearinterpolation" and self.n >= 2:
                # P(xi | xi-1,xi-2) = L1*P(xi) + L2*P(xi|xi-1) + L3*P(xi|xi-1,xi-2)
                nkgrams = [tuple([str(w) for w in x[i:i+k]]) for k in range(1, n+1) ]
                nkgcounts = [ self.grams[nkgram] for nkgram in nkgrams ]
                nkprobs = [ nkgcounts[0]/self.unigrams.total() ]
                nkprobs += [ nkgcounts[i]/nkgcounts[i-1] if (nkgcounts[i]>0 and nkgcounts[i-1]>0) else 0 for i in range(1, len(nkgcounts)) ]
                probs = [ self.lambdas[i] * nkprobs[i] for i in range(len(nkprobs)) ]
                prob = sum(probs)
                prob = prob if prob > 0 else 0.0000000000001  # -.-
            
            else:
                # not smoothing, or something else
                # Unseen tokens in dev/test are masked with <unk> tokens
                prob = ngram_count / count if (ngram_count>0 and count>0) else 0.0000000000001

            log_prob = np.log(prob) / np.log(base)
            log_prob_sum += log_prob

        return log_prob_sum

    def generate(self, num_samples: int, max_steps: int, results_file: str):
        """
        Function for generating text using the n-gram model.

        Inputs:
            num_samples (int): How many samples to generate
            max_steps (int): The maximum length of any sampled output
            results_file (str): Where to save the generated examples
        """
        pass
        # no need
        # see https://edstem.org/us/courses/52897/discussion/4488693

    def learn(self, training_data: List[List[Any]]):
        """
        Function for learning n-grams from the provided training data. You may
        add keywords to this function as needed, provided that the default behavior
        is that of an n-gram model without any label smoothing.
        
        Inputs:
            training_data (List[List[Any]]): A list of model inputs, which should each be lists
                                             of input tokens
        """
        n = self.n
        
        for s1 in training_data:
            # insert as many left-padding tokens as needed
            s = [*s1]
            for i in range(n - 1):
                s.insert(0, self.leftpdx)

            for i in range(len(s) - n + 1):
                ngram = tuple([str(w) for w in s[i:i+n]])
                self.grams[ngram] += 1
                if self.smoothing=="laplace" or self.smoothing=="addk":
                    unigram = tuple([ngram[-1]])
                    self.unigrams[unigram] += 1
                # and also count the n-1 grams (and possibly more) (if applicable)
                if self.n >= 2:
                    # count everything n-1...1 if using linear interpolation for smoothing
                    if self.smoothing == "linearinterpolation":
                        for k in range(1, n):
                            nmkgram = tuple([str(w) for w in s[i:i+n-k]])
                            self.grams[nmkgram] += 1
                            if len(nmkgram)==1:
                                self.unigrams[nmkgram] += 1
                    # otherwise just count n-1 gram
                    else:
                        nm1gram = tuple([str(w) for w in s[i:i+n-1]])
                        self.grams[nm1gram] += 1

        # more stuff for smoothing
        if self.smoothing == "label_smooth":
            smm_counts = self.smm() 
            list_change = [(i, v) for k, v in smm_counts.items() for i in self.grams.keys() if i[:-1] == k]
            for i, v in list_change:
                self.grams[i] = (1 - self.smoothing_factor) * (self.grams[i] / v) + (self.smoothing_factor / len(self.grams))
        if self.smoothing == "linearinterpolation":
            self.grams[tuple([self.unk])] += 1
            self.unigrams[tuple([self.unk])] += 1

    def smm(self):
        context_totals = Counter()
        for context, target_count in self.grams.items():
            context_totals[context[:-1]] += target_count
        return context_totals
def main():
    # Get key arguments
    args = get_args()

    # Get the data for language-modeling and WER computation
    tokenization_level = args.tokenization_level
    model_type = "n_gram"
    print("Model type: ", model_type)
    train_data, val_data, dev_data, test_data, dev_wer_data, test_wer_data = load_data(tokenization_level, model_type)

    # Initialize and "train" the n-gram model
    print("~ initializing and training NGramLM model..")
    n = args.n
    sm = args.smoothing
    model = NGramLM(n, sm)
    model.learn(train_data)

    # Evaluate model perplexity
    print("~ evaluating model perplexity..")
    val_perplexity = evaluate_perplexity(model, val_data)
    print(f'Model perplexity on the val set: {val_perplexity}')
    dev_perplexity = evaluate_perplexity(model, dev_data)
    print(f'Model perplexity on the dev set: {dev_perplexity}')
    test_perplexity = evaluate_perplexity(model, test_data)
    print(f'Model perplexity on the test set: {test_perplexity}')    

    # Evaluate model WER
    print("~ evaluating model wer..")
    experiment_name = args.experiment_name
    dev_wer_savepath = os.path.join('results/dev', f'{experiment_name}_n_gram_dev_wer_predictions.csv')
    dev_wer_savepath = os.path.join('results/dev', f'{experiment_name}_n_gram_dev_wer_predictions.csv')
    rerank_sentences_for_wer(model, dev_wer_data, dev_wer_savepath)
    dev_wer = compute_wer('data/wer_data/dev_ground_truths.csv', dev_wer_savepath)
    print("Dev set WER was: ", dev_wer)

    test_wer_savepath = os.path.join('results', f'{experiment_name}_n_gram_test_wer_predictions.csv')
    rerank_sentences_for_wer(model, test_wer_data, test_wer_savepath)

    # # Generate text from the model
    # generation_path = os.path.join('generations', f'{experiment_name}_n_gram_generation_examples.pkl')
    # num_samples = args.num_samples
    # max_steps = args.max_steps
    # model.generate(num_samples, max_steps, generation_path)
    
    print("~ done :)")


def full_experiment():
    import csv, datetime
    model_type = "n_gram"
    tokenlevels = ["character", "subword", "word"]
    n_values = [1, 2, 3]
    smoothingfunctions = ["none", "laplace", "linearinterpolation", "idkwhatthisis"]
    
    for tokenization_level in tokenlevels:
        train_data, val_data, dev_data, test_data, dev_wer_data, test_wer_data = load_data(tokenization_level, model_type)
        for n in n_values:
            for sm in smoothingfunctions:
                # skip some cmbinations
                if (sm=="linearinterpolation" and n==1):
                    continue
                
                print(f"~ running experiment - m:{model_type} tkl:{tokenization_level} n:{n} sm:{sm}")
                # Initialize and "train" the n-gram model
                print("~   training NGramLM model..")
                model = NGramLM(n, sm)
                model.learn(train_data)
                # Evaluate model perplexity
                print("~   evaluating model perplexity..")
                val_perplexity = evaluate_perplexity(model, val_data)
                print(f'    Model perplexity on the val set: {val_perplexity}')
                dev_perplexity = evaluate_perplexity(model, dev_data)
                print(f'    Model perplexity on the dev set: {dev_perplexity}')
                test_perplexity = evaluate_perplexity(model, test_data)
                print(f'    Model perplexity on the test set: {test_perplexity}')
                # Evaluate model WER
                print("~   evaluating model wer..")
                experiment_name = "exp"
                dev_wer_savepath = os.path.join('results/dev', f'{experiment_name}_n_gram_dev_wer_predictions.csv')
                rerank_sentences_for_wer(model, dev_wer_data, dev_wer_savepath)
                dev_wer = compute_wer('data/wer_data/dev_ground_truths.csv', dev_wer_savepath)
                print("    Dev set WER was: ", dev_wer)
                # log experiment
                with open("./testrun_logs.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    experiment_name = f"e_{datetime.datetime.now().replace(microsecond=0).isoformat(' ')}"
                    vocabsize = len(model.grams.keys())
                    writer.writerow([experiment_name, model_type, tokenization_level, n, sm,
                                     vocabsize, val_perplexity, dev_perplexity, test_perplexity, dev_wer])
                print("~   done :)")


if __name__ == "__main__":
    args = get_args()
    if args.mode and args.mode == "full_experiment" and "full_experiment" in dir():
        print("\n~ running full experiment mode, this will take a while :) ...\n")
        full_experiment()
    else:
        main()
