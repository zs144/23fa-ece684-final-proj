## Plan
### Data
- BBC News dataset from Kaggle ([link](https://www.kaggle.com/datasets/pariza/bbc-news-summary/data))
- CNN-DailyMail News dataset from Kaggle ([link](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail))

### Methods
- Method 1 (baseline): extractive + frequency-based sentence scoring
- Method 2: extractive + DL-based methods (there are some examples in the "Code" section of the two above Kaggle pages)
- Method 3: TODO

### Evaluation
- Metric 1: ROUGE
- Metric 2: TODO

## Useful Resources

### End-to-End Hands-on Guide
- A notebook on Kaggle ([link](https://www.kaggle.com/code/patelris/summarizing-medical-documents)) goes through the basic processing of document summarization using frequency-based sentence scoring, a simple probablistic approch without any complicated ML/DL techniques.

- An intro-level tutorial article ([link](https://www.analyticsvidhya.com/blog/2021/11/a-beginners-guide-to-understanding-text-summarization-with-nlp/)) on how to implement a text summarizer in Python. The article provides two approch: (1) using a pre-trained summarizer; (2) implement a simple summarizer using frequency-based sentence scoring.

- A Jyputer notebook ([link](https://github.com/aravindpai/How-to-build-own-text-summarizer-using-deep-learning/blob/master/How_to_build_own_text_summarizer_using_deep_learning.ipynb)) serving as a guide for developing a DL-based document summarizer from scratch using `Keras`.

- A HuggingFace tutorial ([link](https://huggingface.co/learn/nlp-course/chapter7/5?fw=pt#fine-tuning-mt5-with-accelerate
)) on summarization with every step included (and `PyTorch` / `TensorFlow` implementations!).


### Models Overview
- An article on "5 Powerful Text Summarization Techniques in Python" ([link](https://www.turing.com/kb/5-powerful-text-summarization-techniques-in-python)).

- PapersWithCode leaderboard ([link](https://paperswithcode.com/sota/document-summarization-on-cnn-daily-mail)) for models (mostly **DL-based** models) and their performance (ROUGE score) on CNN-DailyMail News dataset.


### Evaluation Metrics
- An Medium article ([link](https://medium.com/nlplanet/two-minutes-nlp-learn-the-rouge-metric-by-examples-f179cc285499)) explaining what is the ROUGE metric, a measurement commonly used to evaluate how well the generated summary is matched with the ground truth. It also mentions BLUE metric in the companion article ([link](https://medium.com/nlplanet/two-minutes-nlp-learn-the-bleu-metric-by-examples-df015ca73a86)) and compare the two. [**Comments**: I think we can try both later.]

- An Medium article ([link](https://fabianofalcao.medium.com/metrics-for-evaluating-summarization-of-texts-performed-by-transformers-how-to-evaluate-the-b3ce68a309c3)) going through some most popular metrics used to evaluate the quality of summaries, including ROUGE, BLEU, BERTScore, and METEOR. Python packages and functions to compute these metrics are provided at the end of each metric's section. [**Comments**: we need consider both syntactical and semantics matches]

- The source code repo ([link](https://github.com/Tiiiger/bert_score)) for BERTScore.

- An official tutorial ([link](https://github.com/Tiiiger/bert_score/blob/master/example/Demo.ipynb)) to demo `bert_score` package.

- A short note ([link](https://github.com/Tiiiger/bert_score/blob/master/journal/rescale_baseline.md)) elaborating on the rescaling the BERTScore.

- An interface ([link](https://huggingface.co/spaces/evaluate-metric/bertscore)) to compute BERTScore provided by HuggingFace. [**Comments**: Although we use `bert_score` package in our code, this interface is still useful as it provides a convenient GUI, plus some short analysis.]
