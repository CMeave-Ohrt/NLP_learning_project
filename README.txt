------------NLP Learning Project---------------

The goal of this project is to build various natural language processing modules from scratch using PyTorch. Eventually, I would like to implement a German - English translator.

So far I have:
-Implemented a language corpus that can read text and generate an appropriate vocabulary
-Implemented word vectorization using SkipGram
-Implemented a crude Sentiment Analyzer based on Logistic Regression on word vector sums
-Implemented a more sophisticated Sentiment Analyzer using a bidirectional GRU
-Implemented a text generator using a multilayer LSTM

The files are split over two projects:
-IMDB contains files for Sentiment Analysis on the IMDB review dataset
-Bible contains files used for text-generation of bible-like texts

The following files and folders are contained in here:
-Tokenization: Contains word corpus class and creates my_corpus when run
-Vectorization: Contains vectorizer class and all its training methods
-Sentiment_Analysis: contains sentiment analyzer and all its training methods
-Vectorization_run: Runs vectorization training on data set
-Training_Sentiment_Analysis: Trains sentiment analysis on data
-Testing: Tests sentiment analysis
-Text_Generator: Implements a class TextGenerator
-Text_Generator_run: runs and trains the text generator

-Parameters: Contains various saved parameters
-Data: Contains Bible.txt for Bible folder. For IMDB Should contain test and training set of 50k each IMDB reviews. Download data from https://ai.stanford.edu/~amaas/data/sentiment/. Data from Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011)