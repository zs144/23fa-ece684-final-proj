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
    data_path = f'/home/ld258/projects/'

    save_path = f'/home/ld258/projects/{model_name}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    df_test = load_dataset(f'{data_path}/test_llm.csv')
    logging.info(f"Number of records in test set: {len(df_test)}" )

    df = pd.DataFrame(columns=['id', 'pred_summary'])
    df.set_index('id', inplace=True)
    rouge = Rouge()

    for i in range(len(df_test)):
        start_time = time.time()
        current_id = df_test['id'][i]
        summary = summarize_llm(df_test['article'][i], model_name, model_path = model_path, trained_locally = trained_locally)
        df.loc[current_id] = summary
        # logging.info(f"Inference and score computation for single text: {time.time() - start_time:.2f} seconds")

        df.to_csv(f"{save_path}/{model_name}_trained_locally_{trained_locally}_medical_summary.csv")





