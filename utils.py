
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
nltk.download("stopwords")
nltk.download('punkt')
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration




def summarize_llm_finetuned_model(text, model_name):
    """ Returns the summary of the text using the pre-trained transformer model.
    Parameters :
        - text (str): a string of text needs to be summarized.
        - model_name (str): name of the pre-trained transformer model.
    Returns:
        summary (str): a string of summary
    """

    if 'bart' in model_name.lower():
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)

        # Encode the text and generate summary
        inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
        summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)

        # Decode and print the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary



def summarize_llm(text, model_name, model_path = None, trained_locally = False):
    """ Returns the summary of the text using the pre-trained transformer model.
    Parameters :
        - text (str): a string of text needs to be summarized.
        - model_name (str): name of the pre-trained transformer model.
    Returns:
        summary (str): a string of summary
    """

     # Initialize tokenizer and model based on the given model name
    if 't5' in model_name.lower():
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        if trained_locally:
            model = T5ForConditionalGeneration.from_pretrained(model_path)
        else:
            model = T5ForConditionalGeneration.from_pretrained(model_name)

    elif 'bart' in model_name.lower():
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)


    else:
        # Default to using Auto classes for tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Read dataset.
def load_dataset(dataset_path: str) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    # df.drop(columns=['id'], inplace=True) # Drop id column
    df.dropna(inplace=True) # Drop null values (if any)
    return df


def summarize(text: str, summary_len: int) -> str:
    """ Extract sentences from text as the summary using TF-IDF scoring.

    Parameters:
        - text (str): a string of text needs to be summarized.
        - summary_len (int): number of sentences in the summary.

    Returns:
        summary (str): a string of summary

    Notes:
        The score for each sentence is the average TF-IDF score of words (tokens)
        whose score is not zero.
    """
    # Initialize a TF-IDF Vectorizer.
    stop_words = stopwords.words('english')
    tfidf = TfidfVectorizer(stop_words=stop_words, norm='l1')
    # Tokenize sentences (ie, split text into individual sentences).
    sents = nltk.tokenize.sent_tokenize(text)
    # Remove overly short sentences.
    sent_lens = [len(sent) for sent in sents]
    avg_sent_len = sum(sent_lens) / len(sent_lens)
    sents = [sent for sent in sents if len(sent) > avg_sent_len * 0.5]
    # Perform TF-IDF.
    X = tfidf.fit_transform(sents)
    # Compute each sentence score.
    scores = np.zeros(len(sents))
    for i in range(len(sents)):
        score = X[i,:][X[i,:] != 0].mean()
        scores[i] = score

    # Sort the scores.
    sort_idx = np.argsort(-scores)
    # Concatenate sentences with top scores as the summary.
    summary = ''
    for i in sort_idx[:summary_len]:
        summary += (sents[i] + ' ')
    return summary