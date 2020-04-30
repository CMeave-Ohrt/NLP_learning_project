------------NLP Learning Project---------------

The goal of this project was to build various natural language processing modules from scratch using PyTorch.

So far I have:
-Implemented a language corpus that can read text and generate an appropriate vocabulary
-Implemented word vectorization using SkipGram
-Implemented a crude Sentiment Analyzer based on Logistic Regression on word vector sums
-Implemented a more sophisticated Sentiment Analyzer using a bidirectional GRU (achieving 89% accuracy)
-Implemented a text generator using a multilayer LSTM and Beam search
-Implemented a sequence to sequence German to English translator using attention, teacher forcing
-Implemented reinforcement learning with policy gradients to further train the translator (achieving .27 BLEU)

The files are split over three projects:
-IMDB contains files for Sentiment Analysis on the IMDB review dataset
-Bible contains files used for text-generation of bible-like texts
-Translator contains files used for a German-to-English translator

The following files can be found in the corresponding directories:
-IMDB:
--Tokenization: Contains word corpus class and creates my_corpus when run
--Vectorization: Contains vectorizer class and all its training methods
--Sentiment_Analysis: contains sentiment analyzer and all its training methods (for both Logistic Regression and GRU)
--Vectorization_run, Training_Sentiment_Analysis, Testing: short scripts to run/test any of the implemented methods
-Training_Sentiment_Analysis: Trains sentiment analysis on data

-Bible:
--Tokenization_for_Textgen, Vectorization: Slightly adapted version of tokenization and vectorization
--Text_Generator: Implements a class TextGenerator based on layered LSTM's: Given a sequence of text, it predicts the next word
--Beaming: Implements beam search to generate the most likely text. Also implements varied beam search to generate text with more variance
--Text_Generator_run, Bible_corpus_creation, Bible_vectorization: short scripts to run any methods


-Translator:
--Beaming, Tokenization_for_translation, Vectorization: Slightly adapted versions, beaming now contains batched beam search
--Translator: Implements translator class with sequence to sequence model, trainable with both teacher forcing and reinforcement learning
-Prepare_Tatobea_Data, Translator_training, Translator_vectorization, Translator_testing: short scripts to run any methods

Further every folder contains saved parameter states and the training/testing data
For IMDB the data folder should test and training set of 50k each IMDB reviews. Download data from https://ai.stanford.edu/~amaas/data/sentiment/. Data from Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. (2011). Learning Word Vectors for Sentiment Analysis. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011)