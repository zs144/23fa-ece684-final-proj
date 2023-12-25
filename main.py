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
from utils import summarize,  load_dataset, summarize_llm
import torch
import time
import os
import logging
logging.basicConfig(level=logging.INFO)

logging.getLogger().setLevel(logging.INFO)

if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    logging.info (f"Current GPU Number: {current_device}")
    logging.info (f"Current GPU Number: {current_device}")
    logging.info (torch.cuda.get_device_name(current_device))
else:
    logging.info ("CUDA is not available. No GPU detected.")


model_name = "t5-small"
data_path = f'/home/ld258/projects/nlp/project/dataset/cnn_dailymail/'
save_path = f'/home/ld258/projects/nlp/project/results/'
if not os.path.exists(save_path):
    os.makedirs(save_path)


# Load train data.
df_train = load_dataset(f'{data_path}/train.csv')
logging.info(f"Number of records in training set: {len(df_train)}" )

df_val = load_dataset(f'{data_path}/validation.csv')
logging.info(f"Number of records in validation set: {len(df_val)}")


df_test = load_dataset(f'{data_path}/test.csv')
logging.info(f"Number of records in test set: {len(df_test)}" )


# Remove redundant newline character ('\n').
df_train['highlights'] = df_train['highlights'].str.replace('\n', ' ', regex=True)
# Remove the extra whitespace before the periods.
df_train['highlights'] = df_train['highlights'].str.replace(' \.','.', regex=False)

df_val['highlights'  ] = df_val['highlights'  ].str.replace('\n', ' ', regex=True)
df_val['highlights'  ] = df_val['highlights'  ].str.replace(' \.','.', regex=False)

df_test['highlights' ] = df_test['highlights' ].str.replace('\n', ' ', regex=True)
df_test['highlights' ] = df_test['highlights' ].str.replace(' \.','.', regex=False)

df = pd.DataFrame(columns=['id', 'actual_summary', 'pred_summary', 'rogue1_f1', 'rogue2_f1', 'rogueL_f1'])
df.set_index('id', inplace=True)
rouge = Rouge()

# rouge_scores = []
for i in range(len(df_test)):
    start_time = time.time()
    current_id = df_test['id'][i]
    hyps_summary = summarize_llm(df_test['article'][i], model_name)
    refs_summary = df_test['highlights'][i]

    rogue_score = rouge.get_scores(hyps=hyps_summary, refs=refs_summary)

    logging.info(f"Current Scores: {rogue_score[0]['rouge-1']['f']}, {rogue_score[0]['rouge-2']['f']}, {rogue_score[0]['rouge-l']['f']}")

    save_data = [refs_summary, hyps_summary, rogue_score[0]['rouge-1']['f'], rogue_score[0]['rouge-2']['f'], rogue_score[0]['rouge-l']['f']]

    df.loc[current_id] = save_data
    logging.info(f"Inference and score computation for single text: {time.time() - start_time:.2f} seconds")

    df.to_csv(f"{save_path}/{model_name}_summarizer.csv")

