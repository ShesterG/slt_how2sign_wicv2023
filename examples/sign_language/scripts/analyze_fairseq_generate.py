from argparse import Namespace
import argparse
import os
from sacremoses import MosesDetokenizer
import truecase
from fairseq.data import encoders
import sacrebleu
from bleurt import score
#from tqdm import tqdm
from fairseq.tasks.sign_to_text import SignToTextConfig, SignToTextTask

from sacrebleu.metrics import BLEU, CHRF, TER
#import pandas as pd

def decode(tokens, bpe_tokenizer):
    '''Decode the output tokens into a decoded string.'''
    if bpe_tokenizer:
        tokens = bpe_tokenizer.decode(tokens)
    #if moses_detok:
        #tokens = moses_detok.detokenize(tokens.split())
        #tokens = truecase.get_true_case(tokens)
    return tokens

def parse_generate_file(generate_file, backlist_file, partition, path_to_vocab, OLD):
    '''Returns all H and T lines found in the generate file, grouped by id'''
    
    config = SignToTextConfig()
    config.bpe_sentencepiece_model = path_to_vocab
    config.data = f"/content/gbucketafrisign/SLT/bilingual_baseline/{OLD}"
    task = SignToTextTask.setup_task(config)
    task.load_dataset(f"cvpr23.fairseq.i3d.{partition}.how2sign")
    bpe_tokenizer = encoders.build_bpe(
            Namespace(
                bpe='sentencepiece',
                sentencepiece_model=config.bpe_sentencepiece_model
            )
    )
    #moses_detok = MosesDetokenizer(lang='en') 

    with open(generate_file, 'r') as file:
        generate_lines = file.read().split("\n")
    with open(backlist_file, 'r') as file:
        blacklisted_words = file.read().split("\n")
    
    dict_generated = {}
    idx=0
    for line in generate_lines:
        #Check if we should skip the line
        if line.startswith('Generate'):
            break
        idx = line.split('\t')[0].split('-')[1]
        if idx not in dict_generated.keys():
            dict_generated[idx] = {}
        if line.startswith('H'):
            pred = line.split('\t')[2]
            pred_processed = decode(pred, bpe_tokenizer)
            # no_pred = [word for word in pred_processed.translate(str.maketrans('', '', '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~')).split(' ') if word not in blacklisted_words]
            # if len(no_pred) == 0:#If the list is empty, we add a space to avoid errors
            #     no_pred.append(' ')
            if idx in dict_generated.keys():
                dict_generated[idx]['pred'] = pred_processed.lower()
                #dict_generated[idx]['no_pred'] = " ".join(no_pred)
                
        elif line.startswith('T'): #We should check this from the dataset, as we want the raw text
            ref = line.split('\t')[1]
            #check from the dataset, this id and see if it is the same but without lowercasing
            #real_ref = task.datasets[f'cvpr23.fairseq.i3d.{partition}.how2sign'].get_label(int(idx))
            #file_rgb = task.datasets[f'cvpr23.fairseq.i3d.{partition}.how2sign'][int(idx)]['vid_id']
            # no_ref= [word for word in real_ref.translate(str.maketrans('', '', '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~')).split(' ') if word not in blacklisted_words]
            if idx in dict_generated.keys():
                dict_generated[idx]['ref'] = ref.lower()
                # dict_generated[idx]['no_ref'] = " ".join(no_ref)
                #dict_generated[idx]['file_rgb'] = file_rgb
                
    return dict_generated

def compute_metrics(dict_generated):
    '''Computes the different metrics following the valid_step of our task'''
    # preds, refs, preds_reduced, refs_reduced = [], [], [], []
    preds, refs = [], []
    for idx in sorted(dict_generated.keys()):
        pred = dict_generated[idx]['pred']
        #no_pred = dict_generated[idx]['no_pred']
        ref = dict_generated[idx]['ref']
        #no_ref = dict_generated[idx]['no_ref']
        bleu = BLEU().corpus_score([pred], [[ref]])
        dict_generated[idx]['bleu'] = round(bleu.score,2)
        # if no_ref == '' or no_pred == '':
        #     continue
        
        #bleu_reduced = BLEU().corpus_score([no_pred], [[no_ref]])
        #dict_generated[idx]['bleu_reduced'] = round(bleu_reduced.score,2)
        
        preds.append(dict_generated[idx]['pred'])
        refs.append(dict_generated[idx]['ref'])
        # preds_reduced.append(dict_generated[idx]['no_pred'])
        # refs_reduced.append(dict_generated[idx]['no_ref'])
    
    bleu = BLEU().corpus_score(preds, [refs])
    
    checkpoint = "/content/bleurt/BLEURT-20"
    references = refs
    candidates = preds
    scorer = score.BleurtScorer(checkpoint)
    scores = scorer.score(references=references, candidates=candidates)
    #assert isinstance(scores, list) and len(scores) == 1
    scores_brt = round(sum(scores)/len(scores),4)*100
    #print(scores_brt)
    

    # Runs the scoring.
    #bleu1 = BLEU(max_ngram_order=1).corpus_score(preds, [refs])
    #bleu2 = BLEU(max_ngram_order=2).corpus_score(preds, [refs])
    #bleu3 = BLEU(max_ngram_order=3).corpus_score(preds, [refs])
    #bleu4 = BLEU(max_ngram_order=4).corpus_score(preds, [refs])
    #reduced_bleu = BLEU().corpus_score(preds_reduced, [refs_reduced])
    
    chrf = CHRF(word_order=2).corpus_score(preds, [refs])
    #reduced_chrf = CHRF(word_order=2).corpus_score(preds_reduced, [refs_reduced])
    
    #print(round(bleu.score,2), scores_brt, round(chrf.score,2))
    #result = f'{round(bleu.score,2)}, {scores_brt}, {round(chrf.score,2)}'
    #type(result)
    #print(result)
    return bleu, scores_brt, chrf
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process text generation output and compute BLEU scores from task generate output.')
    parser.add_argument('--OLD', type=str, required=True, help='Path where the generates are located. Before the experiment/generates/partition folders.')
    parser.add_argument('--vocab', type=str, required=True, help='Path where the vocabulary is stored. Before the /vocab folder')
    parser.add_argument('--NEW', type=str, required=True, help='Experiment name that we want to test.')
    args = parser.parse_args()
    
    generate_file = f'/content/gbucketafrisign/SLT/bilingual_baseline/{args.OLD}/baseline_6_3_dp03_wd_2/generates/{args.NEW}/cvpr23.fairseq.i3d.test.how2sign/checkpoint_best.out'
    blacklist_file = 'scripts/blacklisted_words.txt'
    path_to_vocab = f'/content/gbucketafrisign/SLT/bilingual_baseline/{args.OLD}/vocab/cvpr23.train.how2sign.unigram{args.vocab}_lowercased.model'
    OLD = args.OLD
    print(f'Analyzing file: {generate_file}', flush=True)
    dict_generate = parse_generate_file(generate_file, blacklist_file, "test", path_to_vocab, OLD)
    bleu, scores_brt, chrf = compute_metrics(dict_generate)
    results = f"{args.NEW},{round(bleu.score,2)},{scores_brt},{round(chrf.score,2)}"
    #text = "Hello, world!" 
    filename = "/content/gbucketafrisign/SLT/bilingual_baseline/000Results/B36.txt"
    # Open the file in "append" mode to write the string in a new line
    with open(filename, "a") as file:
        # Write the string followed by a newline character 
        file.write(results + "\n")    
    #print("BLEU: ", bleu)
    print(results)
