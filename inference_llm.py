# Basic packages
import pandas as pd
import numpy as np

# Package for NLP
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")

# Packages for evalution
## ROUGE score
from rouge import Rouge
## BLEU score
from nltk.translate.bleu_score import sentence_bleu
## BERT score
import bert_score
## METEOR score
nltk.download('wordnet')
from nltk.translate.meteor_score import meteor_score
from utils import summarize,  load_dataset, summarize_llm, summarize_llm_finetuned_model
import torch
import time
import os
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

# Check if GPU is available.
if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    logging.info (f"Current GPU Number: {current_device}")
    logging.info (f"Current GPU Number: {current_device}")
    logging.info (torch.cuda.get_device_name(current_device))
else:
    logging.info ("CUDA is not available. No GPU detected.")


if __name__== '__main__'   :

    trained_locally = True
    model_name = "t5-small"   #t5-small "facebook/bart-large-cnn"

    model_path = '/scratch/railabs/ld258/output/summarizer_models/results_x/checkpoint-47000'

    if '/' in model_name:
        save_name = model_name.replace('/', '_')
    else:
        save_name = model_name
    data_path = f'/home/ld258/projects/nlp/project/data/cnn_dailymail/'
    save_path = f'/home/ld258/projects/nlp/project/results/{model_name}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df_test = load_dataset(f'{data_path}/test.csv')
    logging.info(f"Number of records in test set: {len(df_test)}" )

    df_test['highlights' ] = df_test['highlights' ].str.replace('\n', ' ', regex=True)
    df_test['highlights' ] = df_test['highlights' ].str.replace(' \.','.', regex=False)

    df = pd.DataFrame(columns=['id', 'actual_summary', 'pred_summary', 'rogue1_f1', 'rogue2_f1', 'rogueL_f1'])
    df.set_index('id', inplace=True)
    rouge = Rouge()

    for i in range(len(df_test)):
        start_time = time.time()
        current_id = df_test['id'][i]
        if 'cnn' in model_name:
            hyps_summary = summarize_llm_finetuned_model(df_test['article'][i], model_name)
        else:
            hyps_summary = summarize_llm(df_test['article'][i], model_name, model_path = model_path, trained_locally = trained_locally)
        refs_summary = df_test['highlights'][i]

        rogue_score = rouge.get_scores(hyps=hyps_summary, refs=refs_summary)

        logging.info(f"Current Scores: {rogue_score[0]['rouge-1']['f']}, {rogue_score[0]['rouge-2']['f']}, {rogue_score[0]['rouge-l']['f']}")

        save_data = [refs_summary, hyps_summary, rogue_score[0]['rouge-1']['f'], rogue_score[0]['rouge-2']['f'], rogue_score[0]['rouge-l']['f']]

        df.loc[current_id] = save_data
        logging.info(f"Inference and score computation for single text: {time.time() - start_time:.2f} seconds")

        df.to_csv(f"{save_path}/{save_name}_trained_locally_{trained_locally}_summary.csv")










    # # Load train data.
    # df_train = load_dataset(f'{data_path}/train.csv')
    # logging.info(f"Number of records in training set: {len(df_train)}" )

    # df_val = load_dataset(f'{data_path}/validation.csv')
    # logging.info(f"Number of records in validation set: {len(df_val)}")


        # # Remove redundant newline character ('\n').
    # df_train['highlights'] = df_train['highlights'].str.replace('\n', ' ', regex=True)
    # # Remove the extra whitespace before the periods.
    # df_train['highlights'] = df_train['highlights'].str.replace(' \.','.', regex=False)

    # df_val['highlights'  ] = df_val['highlights'  ].str.replace('\n', ' ', regex=True)
    # df_val['highlights'  ] = df_val['highlights'  ].str.replace(' \.','.', regex=False)
