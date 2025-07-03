from typing import List, Any, Tuple
import pandas as pd
import torch
from evaluate import load
import csv
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def rerank_sentences_for_wer(model: Any, wer_data: List[Any], savepath: str, vocab = None):
    """
    Function to rerank candidate sentences in the HUB dataset. For each set of sentences,
    you must assign each sentence a score in the form of the sentence's acoustic score plus
    the sentence's log probability. You should then save the top scoring sentences in a .csv
    file similar to those found in the results directory.

    Inputs:
        model (Any): An n-gram or Transformer model.
        wer_data (List[Any]): Processed data from the HUB dataset. 
        savepath (str): The path to save the csv file pairing sentence set ids and the top ranked sentences.
    """
    # TODO
    if str(type(model)) == "<class '__main__.CharacterLevelTransformer'>":
        top_ranked_sentences = {}
        for s_id, d in wer_data.items():
            sentences = d['sentences']
            sentences  = [[vocab.index(char) if char in vocab else vocab.index('unk') for char in ''.join(s)] for s in sentences]
            sentences = [x[:86] if len(x) > 86 else x + [36]*(86-len(x)) for x in sentences]
            # print(sentences[0]) 
            sentences = torch.tensor(sentences).to(DEVICE)[None,:,:]
            acoustic_scores =torch.tensor(d['acoustic_scores']).to(DEVICE)
            log_probabilities = []
            for s in sentences:
                log_prob = model.log_probability(s, s)
                log_probabilities.append(log_prob)
                
            scores = [a + b for a, b in zip(acoustic_scores, log_probabilities)]
            sentence_scores = list(zip(sentences, scores))

            sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
            top_ranked_sentences[s_id] = sorted_sentences[0][0]
        

    else:
        top_ranked_sentences = {}
        for entry in wer_data.items():
            s_id, d = entry
            sentences = d['sentences']
            acoustic_scores = d['acoustic_scores']
            
            log_probabilities = []
            for s in sentences:
                log_prob = model.log_probability(s) 
                log_probabilities.append(log_prob)
                
            scores = [a + b for a, b in zip(acoustic_scores, log_probabilities)]
            sentence_scores = list(zip(sentences, scores))

            sorted_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)
            top_ranked_sentences[s_id] = sorted_sentences[0][0]

    # Save top ranked sentences to CSV file
    with open(savepath, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'sentences']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for sentence_id, sentence in top_ranked_sentences.items():
            writer.writerow({'id': sentence_id, 'sentences': sentence})


def compute_wer(gt_path, model_path):
    # Load the sentences
    ground_truths = pd.read_csv(gt_path)['sentences'].tolist()
    guesses = pd.read_csv(model_path)['sentences'].tolist()

    # Compute wer
    wer = load("wer")
    wer_value = wer.compute(predictions=guesses, references=ground_truths)
    return wer_value