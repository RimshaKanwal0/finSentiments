import os
import psutil
import platform
import scipy.sparse as sp
from scipy.sparse import csr_matrix, hstack
# import joblib
# from itertools import permutations
from itertools import combinations, chain
import numpy as np
import pandas as pd
import re
import pytz # timezone
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import multiprocessing
from multiprocessing import Process, Queue
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
# from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
# from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from imblearn.over_sampling import SMOTE
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
#!pip install num2words
from num2words import num2words

# Load dataset
df = pd.read_csv('E:/django/sentimentproject/finSentiments/dataset/data.csv')

# Custom financial domain-specific stopwords
custom_stopwords = set(["stock", "price", "earnings", "report", "investors", "company"])
# Function to convert numbers to text representations
def convert_numbers_to_text(tokens):
    converted_tokens = []
    for token in tokens:
        if re.match(r'^-?\d+(?:\.\d+)?$', token):
            converted_token = num2words(token, to='currency')
            converted_tokens.extend(converted_token.split())
        else:
            converted_tokens.append(token)
    return converted_tokens

# Tokenization and preprocessing function
def preprocess_text(text, remove_stopwords=False, lemmatization=False, custom_stopwords=[]):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    if lemmatization:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if word not in custom_stopwords]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = convert_numbers_to_text(tokens)
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text
df['Preprocessed_Sentence'] = df['Sentence'].apply(preprocess_text, remove_stopwords=True, lemmatization=True)
preprocessed_sentences = df['Preprocessed_Sentence']


def bytes_to_mb_or_gb(byte_value):
    if byte_value >= 1024**3:  # GB
        return f"{byte_value / (1024**3):.2f} GB"
    elif byte_value >= 1024**2:  # MB
        return f"{byte_value / (1024**2):.2f} MB"
    else:  # bytes
        return f"{byte_value} bytes"

# Main function for training and evaluating blended classifiers
# def train_and_evaluate_blended_classifier(
#     classifier_combo, classifiers, X_train, X_test, y_train, y_test,
#     additional_info, X_train_tfidf_dense, X_test_tfidf_dense
# ):
#     print(f"Blending the following classifiers: {classifier_combo}")
#     start_time = time.time()
#     num_cores = multiprocessing.cpu_count()
#     processor_type = platform.processor()
#     os_name = platform.system() + '(' + platform.release() + ')'
#     ram_total = psutil.virtual_memory().total
#     ram_total_mb = ram_total / (1024**2)
#     cpu_before = psutil.cpu_percent()
#     available_memory_before = psutil.virtual_memory().available
#     # Create the voting classifier with the selected combination
#     selected_classifiers = [(name, classifiers[name]) for name in classifier_combo]
#     voting_classifier = VotingClassifier(estimators=selected_classifiers, voting='hard')
#     # Fit and evaluate the classifier
#     voting_classifier.fit(X_train_tfidf_dense, y_train)
#     final_predictions = voting_classifier.predict(X_test_tfidf_dense)
#     accuracy = accuracy_score(y_test, final_predictions)
#     precision = precision_score(y_test, final_predictions, average='weighted', zero_division=1)
#     recall = recall_score(y_test, final_predictions, average='weighted')
#     f1 = f1_score(y_test, final_predictions, average='weighted')
#     cm = confusion_matrix(y_test, final_predictions)
#     kappa = cohen_kappa_score(y_test, final_predictions)
#     end_time = time.time()
#     execution_time = end_time - start_time
#     cpu_after = psutil.cpu_percent()
#     available_memory_after = psutil.virtual_memory().available
#     cpu_usage = max(cpu_after - cpu_before, 0)
#     memory_usage = max(available_memory_before - available_memory_after, 0)  # Ensure non-negative
#     memory_usage_str = bytes_to_mb_or_gb(memory_usage)
#
#     results_dict = {
#         'Blended Classifiers': classifier_combo,
#         'Accuracy': accuracy,
#         'Kappa': kappa,
#         'Precision': precision,
#         'Recall': recall,
#         'F1-Score': f1,
#         'Confusion Matrix': cm.tolist(),  # Convert numpy array to list for JSON serialization
#         'Execution Time (s)': execution_time,
#         'Total Classifiers': additional_info.get('Total Classifiers', 'N/A'),
#         'Total Features': additional_info.get('Total Features', 'N/A'),
#         'Training Data Size': additional_info.get('Training Data Size', 'N/A'),
#         'Test Data Size': additional_info.get('Test Data Size', 'N/A'),
#         'Random State': additional_info.get('Random State', 'N/A'),
#         'Preprocessing': additional_info.get('Preprocessing', 'N/A'),
#         'SMOTE': additional_info.get('SMOTE', 'N/A'),
#         'Total CPU Cores': num_cores,
#         'CPU Usage (%)': cpu_usage,
#         'Total RAM (MB)': ram_total_mb,
#         'Memory Usage': memory_usage_str,
#         'Processor Type': processor_type,
#         'OS': os_name
#     }
#     return results_dict
#
# #....
# def train_word2vec_model(df, preprocess_function, model_path="word2vec_model.bin"):
#     df['Preprocessed_Sentence'] = df['Sentence'].apply(preprocess_function)
#     corpus = df['Preprocessed_Sentence'].apply(lambda x: x.split())
#     word2vec_model = Word2Vec(sentences=corpus, vector_size=300, window=5, min_count=1, sg=0, epochs=10)
#     latest_loss = word2vec_model.get_latest_training_loss()
#     vocabulary_words = word2vec_model.wv.index_to_key
#     word2vec_model.save(model_path)
#     return {
#         "latest_loss": latest_loss,
#         "vocabulary_words": vocabulary_words,
#         "model_path": model_path
#     }
# #....
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import LabelEncoder
# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import train_test_split
# import pandas as pd
# def preprocess_and_split_data(df, preprocessing_flag=1, smote_flag=1, test_size=0.2, rand_state=42,):
#     # Apply data preprocessing
#     df['Preprocessed_Sentence'] = df['Sentence'].apply(preprocess_text) if preprocessing_flag == 1 else df['Sentence']
#     # Define features and target
#     X = df['Preprocessed_Sentence']
#     y = df['Sentiment']
#     # Create TF-IDF vectorizer
#     tfidf_vectorizer = TfidfVectorizer()
#     X_tfidf = tfidf_vectorizer.fit_transform(X)
#     label_encoder = LabelEncoder()
#     y_encoded = label_encoder.fit_transform(y)
#     # Apply SMOTE if required
#     if smote_flag == 1:
#         smote = SMOTE(sampling_strategy='auto', random_state=rand_state)
#         X_balanced, y_balanced = smote.fit_resample(X_tfidf, y_encoded)
#     else:
#         X_balanced, y_balanced = X_tfidf, y_encoded
#
#     # Split the dataset into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=test_size,
#                                                         random_state=rand_state)
#
#     # Convert to dense format for further processing
#     X_train_dense = X_train.toarray()
#     X_test_dense = X_test.toarray()
#
#     return X_train_dense, X_test_dense, y_train, y_test, tfidf_vectorizer, label_encoder
#
# #.....
# # Constants
# PREPROCESSING_FLAG = 1
# SMOTE_FLAG = 1
# TEST_SIZE = 0.2
# RAND_STATE = 42
#
# # Preprocess the text data
# df['Preprocessed_Sentence'] = df['Sentence'].apply(preprocess_text)
# # Select features based on the preprocessing flag
# X = df['Preprocessed_Sentence'] if PREPROCESSING_FLAG == 1 else df['Sentence']
# y = df['Sentiment']
# # Create a TF-IDF vectorizer and apply it to the text data
# tfidf_vectorizer = TfidfVectorizer()
# X_train_tfidf = tfidf_vectorizer.fit_transform(X)
# # Encode the target labels
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
# # Create Word2Vec features
# word2vec_features = []
# for doc in df['Preprocessed_Sentence']:
#     tokens = doc.split()
#     word2vec_embeddings = [loaded_model.wv[token] for token in tokens if token in loaded_model.wv]
#     # Calculate the average Word2Vec embedding for the document
#     avg_embedding = np.mean(word2vec_embeddings, axis=0) if word2vec_embeddings else np.zeros(loaded_model.vector_size)
#     word2vec_features.append(avg_embedding)
# # Convert Word2Vec features to a numpy array
# X_word2vec = np.array(word2vec_features)
# # Combine TF-IDF and Word2Vec features
# X_combined = np.hstack((X_train_tfidf.toarray(), X_word2vec))
# # Apply SMOTE if needed
# if SMOTE_FLAG == 1:
#     smote = SMOTE(sampling_strategy='auto', random_state=RAND_STATE)
#     X_train_balanced, y_train_balanced = smote.fit_resample(X_combined, y_encoded)
# else:
#     X_train_balanced, y_train_balanced = X_combined, y_encoded
# # Split the data into training and testing sets
# X_train_tfidf_dense, X_test_tfidf_dense, y_train, y_test = train_test_split(
#     X_train_balanced, y_train_balanced, test_size=TEST_SIZE, random_state=RAND_STATE)
# # Check the shapes of the resulting datasets
# #print(f"Training data shape: {X_train_tfidf_dense.shape}")
# #print(f"Test data shape: {X_test_tfidf_dense.shape}")
#
# #...
# individual_mode = False
#
# classifiers = {
#     #'Logistic Regression': LogisticRegression(max_iter=1000),
#     'SDG': SGDClassifier(),
#     'Support Vector Machine': SVC(),
# }
#
# # Initialize blended_results_list
# results_list = []
# # Initialize blended_results_dict
# results_dict = {}
# # Initialize an empty list to store the rows of the results
# results_rows = []
# # Additional Information
# # Additional information about the code
# additional_info = {
#     'Total Classifiers': len(classifiers),
#     'Total Features': len(tfidf_vectorizer.get_feature_names_out()),
#     'Total Features': X_train_tfidf.shape[1] + X_word2vec.shape[1],
#     'Training Data Size': len(y_train),
#     'Test Data Size': len(X_test_tfidf_dense),
#     'Random State': RAND_STATE,  # Replace with the actual random_state value used in train_test_split
#     'Preprocessing': PREPROCESSING_FLAG,
#     'SMOTE': SMOTE_FLAG
# }




#sir ka code..

# # Load dataset
# df = pd.read_csv('E:/django/sentimentproject/finSentiments/dataset/data.csv')
# # Train a SentencePiece model for Tokenization
# # !pip install sentencepiece
# import sentencepiece as spm
# spm.SentencePieceTrainer.train(input= df, model_prefix='mymodel', vocab_size=10000) # vocab_size=5000
# sp = spm.SentencePieceProcessor(model_file='mymodel.model')

#------take after the dataset
# custom_stopwords_df = pd.read_csv("/content/Drive/MyDrive/python/financial_sentiment_analysis/custom_stopwords.csv")  # StopWord List
#
# A Python module that will load the Loughran-McDonald Master and its components (and optionally separate sentiment dictionaries).
#
# But this technique doesn't worked well in our case. It only added few new features.
#
# """
# Routine to load master dicitonary
# Version for LM 2020 Master Dictionary
#
# Bill McDonald
# Date: 201510 Updated: 202201
# """

# def load_masterdictionary(file_path, print_flag=False, f_log=None, get_other=False):
#     start_local = dt.datetime.now()
#     # Setup dictionaries
#     _master_dictionary = {}
#     _sentiment_categories = ['negative', 'positive', 'uncertainty', 'litigious',
#                              'strong_modal', 'weak_modal', 'constraining']
#     _sentiment_dictionaries = dict()
#     for sentiment in _sentiment_categories:
#         _sentiment_dictionaries[sentiment] = dict()
#
#     # Load slightly modified common stopwords.
#     # Dropped from traditional: A, I, S, T, DON, WILL, AGAINST
#     # Added: AMONG
#     _stopwords = ['ME', 'MY', 'MYSELF', 'WE', 'OUR', 'OURS', 'OURSELVES', 'YOU', 'YOUR', 'YOURS',
#                   'YOURSELF', 'YOURSELVES', 'HE', 'HIM', 'HIS', 'HIMSELF', 'SHE', 'HER', 'HERS', 'HERSELF',
#                   'IT', 'ITS', 'ITSELF', 'THEY', 'THEM', 'THEIR', 'THEIRS', 'THEMSELVES', 'WHAT', 'WHICH',
#                   'WHO', 'WHOM', 'THIS', 'THAT', 'THESE', 'THOSE', 'AM', 'IS', 'ARE', 'WAS', 'WERE', 'BE',
#                   'BEEN', 'BEING', 'HAVE', 'HAS', 'HAD', 'HAVING', 'DO', 'DOES', 'DID', 'DOING', 'AN',
#                   'THE', 'AND', 'BUT', 'IF', 'OR', 'BECAUSE', 'AS', 'UNTIL', 'WHILE', 'OF', 'AT', 'BY',
#                   'FOR', 'WITH', 'ABOUT', 'BETWEEN', 'INTO', 'THROUGH', 'DURING', 'BEFORE',
#                   'AFTER', 'ABOVE', 'BELOW', 'TO', 'FROM', 'UP', 'DOWN', 'IN', 'OUT', 'ON', 'OFF', 'OVER',
#                   'UNDER', 'AGAIN', 'FURTHER', 'THEN', 'ONCE', 'HERE', 'THERE', 'WHEN', 'WHERE', 'WHY',
#                   'HOW', 'ALL', 'ANY', 'BOTH', 'EACH', 'FEW', 'MORE', 'MOST', 'OTHER', 'SOME', 'SUCH',
#                   'NO', 'NOR', 'NOT', 'ONLY', 'OWN', 'SAME', 'SO', 'THAN', 'TOO', 'VERY', 'CAN',
#                   'JUST', 'SHOULD', 'NOW', 'AMONG']
#
#     # Loop thru words and load dictionaries
#     with open(file_path) as f:
#         _total_documents = 0
#         _md_header = f.readline()  # Consume header line
#         print()
#         for line in f:
#             cols = line.rstrip('\n').split(',')
#             word = cols[0]
#             _master_dictionary[word] = MasterDictionary(cols, _stopwords)
#             for sentiment in _sentiment_categories:
#                 if getattr(_master_dictionary[word], sentiment):
#                     _sentiment_dictionaries[sentiment][word] = 0
#             _total_documents += _master_dictionary[cols[0]].doc_count
#             if len(_master_dictionary) % 5000 == 0 and print_flag:
#                 print(f'\r ...Loading Master Dictionary {len(_master_dictionary):,}', end='', flush=True)
#
#     if print_flag:
#         print('\r', end='')  # clear line
#         print(f'\nMaster Dictionary loaded from file:\n  {file_path}\n')
#         print(f'  master_dictionary has {len(_master_dictionary):,} words.\n')
#
#     if f_log:
#         try:
#             f_log.write('\n\n  FUNCTION: load_masterdictionary' +
#                         '(file_path, print_flag, f_log, get_other)\n')
#             f_log.write(f'\n    file_path  = {file_path}')
#             f_log.write(f'\n    print_flag = {print_flag}')
#             f_log.write(f'\n    f_log      = {f_log.name}')
#             f_log.write(f'\n    get_other  = {get_other}')
#             f_log.write(f'\n\n    {len(_master_dictionary):,} words loaded in master_dictionary.\n')
#             f_log.write(f'\n    Sentiment:')
#             for sentiment in _sentiment_categories:
#                 f_log.write(f'\n      {sentiment:13}: {len(_sentiment_dictionaries[sentiment]):8,}')
#             f_log.write(f'\n\n  END FUNCTION: load_masterdictionary: {(dt.datetime.now()-start_local)}')
#         except Exception as e:
#             print('Log file in load_masterdictionary is not available for writing')
#             print(f'Error = {e}')
#
#     if get_other:
#         return _master_dictionary, _md_header, _sentiment_categories, _sentiment_dictionaries, _stopwords, _total_documents
#     else:
#         return _master_dictionary
#
#
# class MasterDictionary:
#     def __init__(self, cols, _stopwords):
#         for ptr, col in enumerate(cols):
#             if col == '':
#                 cols[ptr] = '0'
#         try:
#             self.word = cols[0].upper()
#             self.sequence_number = int(cols[1])
#             self.word_count = int(cols[2])
#             self.word_proportion = float(cols[3])
#             self.average_proportion = float(cols[4])
#             self.std_dev_prop = float(cols[5])
#             self.doc_count = int(cols[6])
#             self.negative = int(cols[7])
#             self.positive = int(cols[8])
#             self.uncertainty = int(cols[9])
#             self.litigious = int(cols[10])
#             self.strong_modal = int(cols[11])
#             self.weak_modal = int(cols[12])
#             self.constraining = int(cols[13])
#             self.syllables = int(cols[14])
#             self.source = cols[15]
#             if self.word in _stopwords:
#                 self.stopword = True
#             else:
#                 self.stopword = False
#         except:
#             print('ERROR in class MasterDictionary')
#             print(f'word = {cols[0]} : seqnum = {cols[1]}')
#             quit()
#         return
#

# #
# # Calling the method to load Loughran-McDonald Master Dictionary
# # if __name__ == '__main__':
# #     start = dt.datetime.now()
# #     print(f'\n\n{start.strftime("%c")}\nPROGRAM NAME: {sys.argv[0]}\n')
# #     f_log = open('/content/Drive/MyDrive/python/financial_sentiment_analysis/Load_MD_Logfile.txt', 'w')
# #     md = (r"/content/Drive/MyDrive/python/financial_sentiment_analysis/Loughran-McDonald_MasterDictionary_1993-2021.csv")
# #     master_dictionary, md_header, sentiment_categories, sentiment_dictionaries, stopwords, total_documents = \
# #         load_masterdictionary(md, True, f_log, True)
# #     print(f'\n\nRuntime: {(dt.datetime.now()-start)}')
# #     print(f'\nNormal termination.\n{dt.datetime.now().strftime("%c")}\n')
# #
#
#
# # Subword tokenization using libraries like SentencePiece or Byte Pair Encoding (BPE) is useful when you want to tokenize text into subword units. This can be particularly helpful for languages with complex morphology or when dealing with out-of-vocabulary words. Here's how you can use these libraries:
# #
# # Using SentencePiece:
# # SentencePiece is a popular subword tokenization library developed by Google. It's used for a wide range of natural language processing tasks.
# #
# # But this technique doesn't worked in our case. Accuracy and number of features fell dramatically, as soon as we applied it.
#
#
#
# # Train a SentencePiece model for Tokenization
# # !pip install sentencepiece
# import sentencepiece as spm
# spm.SentencePieceTrainer.train(input= df, model_prefix='mymodel', vocab_size=10000) # vocab_size=5000
# sp = spm.SentencePieceProcessor(model_file='mymodel.model')
#
#
#
# # The following cell is dedicated for preprocessing and tokenization processes.
# # Custom financial domain-specific stopwords
# custom_stopwords = set(["stock", "price", "earnings", "report", "investors", "company"])
#
# # Function to convert numbers to text representations
# def convert_numbers_to_text(tokens):
#     converted_tokens = []
#     for token in tokens:
#         # Check if it's a number (integer or decimal)
#         if re.match(r'^-?\d+(?:\.\d+)?$', token):
#             converted_token = num2words(token, to='currency')  # Convert number to text with currency format
#             converted_tokens.extend(converted_token.split())  # Split the converted text
#         else:
#             converted_tokens.append(token)
#     return converted_tokens
#
#
# # Tokenization and preprocessing function
# def preprocess_text(text, remove_stopwords=False, lemmatization=False, custom_stopwords=[]):
#     # Convert text to lowercase
#     text = text.lower()
#
#     # Remove HTML tags and URLs
#     text = re.sub(r"<.*?>", "", text)
#     text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
#
#     # Tokenize text
#     tokens = word_tokenize(text)
#
#     # Initialize lemmatizer
#     lemmatizer = WordNetLemmatizer()
#
#     # Apply optional lemmatization
#     if lemmatization:
#         tokens = [lemmatizer.lemmatize(word) for word in tokens]
#
#     # Apply optional stopwords removal
#     if remove_stopwords:
#         stop_words = set(stopwords.words('english'))
#         tokens = [word for word in tokens if word not in stop_words]
#
#     # Filter out custom stopwords (if any)
#     tokens = [word for word in tokens if word not in custom_stopwords]
#
#     # Filter out single characters and symbols
#     tokens = [word for word in tokens if len(word) > 1]
#
#     # Convert numbers to text representations (if needed)
#     tokens = convert_numbers_to_text(tokens)
#
#     # Join tokens back to form the preprocessed text
#     preprocessed_text = ' '.join(tokens)
#
#     return preprocessed_text
#
# # Example of applying the function to your DataFrame
# df['Preprocessed_Sentence'] = df['Sentence'].apply(preprocess_text, remove_stopwords=True, lemmatization=True)
#
# # Access the 'Preprocessed_Sentence' column
# preprocessed_sentences = df['Preprocessed_Sentence']
#
# # Print the first few preprocessed sentences for verification
# # print(preprocessed_sentences.head())
#
#
# # Methods
# # Function to convert bytes to MB or GB
# def bytes_to_mb_or_gb(byte_value):
#     if byte_value >= 1024**3:  # GB
#         return f"{byte_value / (1024**3):.2f} GB"
#     elif byte_value >= 1024**2:  # MB
#         return f"{byte_value / (1024**2):.2f} MB"
#     else:  # bytes
#         return f"{byte_value} bytes"
#
# def train_and_evaluate_blended_classifier(classifier_combo, additional_info, num_classifiers):
#     print(f"Blending the following classifiers: {classifier_combo}")
#
#     start_time = time.time()  # Record the start time
#     start_time = time.time()
#     # Get CPU information
#     num_cores = multiprocessing.cpu_count()
#     processor_type = platform.processor()
#     # Get OS information
#     os_name = platform.system()+'('+platform.release()+')'
#     # Get RAM information
#     ram_total = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
#     ram_total_mb = ram_total / (1024**2)
#     #Calculate CPU and RAM resources before training
#     cpu_before = psutil.cpu_percent()
#     available_memory_before = psutil.virtual_memory().available
#
# ################################################################################
#     # Create the voting classifier with the selected combination
#     selected_classifiers = [(name, classifiers[name]) for name in classifier_combo]
#     voting_classifier = VotingClassifier(estimators=selected_classifiers, voting='hard')
#
#     # Train the voting classifier
#     voting_classifier.fit(X_train_tfidf_dense, y_train)
#
#     # Make final predictions on the test data using the voting classifier
#     final_predictions = voting_classifier.predict(X_test_tfidf_dense)
#
#     # Calculate evaluation metrics for the blended model
#     accuracy = accuracy_score(y_test, final_predictions)
#     precision = precision_score(y_test, final_predictions, average='weighted', zero_division=1)
#     recall = recall_score(y_test, final_predictions, average='weighted')
#     f1 = f1_score(y_test, final_predictions, average='weighted')
#     cm = confusion_matrix(y_test, final_predictions)
#
#     # Calculate Cohen's kappa
#     kappa = cohen_kappa_score(y_test, final_predictions)
#
# ################################################################################
#
#     end_time = time.time()  # Record the end time
#     execution_time = end_time - start_time  # Calculate the execution time
#     # Calculate CPU and RAM resources after training
#     cpu_after = psutil.cpu_percent()
#     available_memory_after = psutil.virtual_memory().available
#
#     # Calculate resource usage during this iteration
#     cpu_usage = max(cpu_after - cpu_before, 0)
#     memory_usage = max(available_memory_after - available_memory_before, 0)  # Ensure non-negative value
#     memory_usage_str = bytes_to_mb_or_gb(memory_usage)
#
#     # print(f"CPU Usage: {cpu_usage:.2f}%")
#     # print(f"Memory Usage: {memory_usage_str}")
#
#     # Store the results in the list as a dictionary
#     results_dict = {
#         'Blended Classifiers': classifier_combo,
#         'Accuracy': accuracy,
#         'Kappa': kappa,
#         'Precision': precision,
#         'Recall': recall,
#         'F1-Score': f1,
#         'Confusion Matrix': cm,
#         'Execution Time (s)': execution_time,
#         'Total Classifiers': additional_info['Total Classifiers'],
#         'Total Features': additional_info['Total Features'],
#         'Training Data Size': additional_info['Training Data Size'],
#         'Test Data Size': additional_info['Test Data Size'],
#         'Random State': additional_info['Random State'],
#         'Preprocessing': additional_info['Preprocessing'],
#         'SMOTE': additional_info['SMOTE'],
#         'Total CPU Cores': num_cores,
#         'CPU Usage (%)': cpu_usage,
#         'Total RAM': ram_total_mb,
#         'Memory Usage': memory_usage_str,
#         'Processor Type': processor_type,
#         'OS': os_name
#     }
#     print(results_dict)
#
#
#     return results_dict
#
#
#
# # Word2Vec:
# # We can use Word2Vec in our code to learn word embeddings from your text data. To do this, we'll need to use a library like Gensim in Python, which provides an easy way to train Word2Vec models.
#
# # Generating our own word2vec model
#
# from gensim.models import Word2Vec, KeyedVectors
# df['Preprocessed_Sentence'] = df['Sentence'].apply(preprocess_text)
# # print(df.columns)
# corpus = df['Preprocessed_Sentence'].apply(lambda x: x.split())  # Assuming each sentence is a space-separated list of words
#
# # Train Word2Vec model with more epochs
# word2vec_model = Word2Vec(sentences=corpus, vector_size=300, window=5, min_count=1, sg=0, epochs=10)
#
# # Monitor Loss: You can track the loss (negative log likelihood) during training using the get_latest_training_loss() method of the Word2Vec model. This method returns the loss for the most recent batch of training data:
# # Check the latest training loss
# latest_loss = word2vec_model.get_latest_training_loss()
# # print(f"Latest Training Loss: {latest_loss}")
# # The loss should decrease over time during training. If it remains constant or increases significantly, it may indicate an issue with your data or training parameters.
#
# # Get the list of words in your Word2Vec model's vocabulary
# vocabulary_words = word2vec_model.wv.index_to_key
# # Print the first 10 words as an example
# # print("First 10 words in the vocabulary:")
# # print(vocabulary_words[:10])
#
# # Test Word Similarity: After training, you can test whether the model has learned meaningful word embeddings by checking word similarity. Gensim's wv attribute provides access to the word vectors. You can use the similarity method to check the similarity between words:
# # Check word similarity between two specific words
# # word1 = "the"
# # word2 = "and"
# # similarity_score = word2vec_model.wv.similarity(word1, word2)
# # print(f"Similarity between '{word1}' and '{word2}': {similarity_score}")
# # Replace "word1" and "word2" with words from your vocabulary to check their similarity. Higher similarity scores indicate that the model has learned meaningful word representations.
#
# # Save the model to a file
# word2vec_model.save("word2vec_model.bin")
#
# # Load the model from the file
# loaded_model = Word2Vec.load("word2vec_model.bin")
#
# loaded_model
#
# # Save the trained model to a local file
# # word2vec_model.save('word2vec_model.bin')
#
# # import shutil
#
# # # Specify the destination folder in your Google Drive
# # destination_folder = '/content/Drive/MyDrive/python/financial_sentiment_analysis/'
#
# # # Move the trained model to Google Drive
# # shutil.move('word2vec_model.bin', destination_folder)
#
#
#
# # Features Extracted from TF-IDF
# # Vectorizing, Balancing, Extracting Features and Splitting the Data.
# # Constants
# PREPROCESSING_FLAG = 1
# SMOTE_FLAG = 1
# TEST_SIZE = 0.2
# RAND_STATE = 42
#
# # Apply data preprocessing to the 'Sentence' column and create 'Preprocessed_Sentence' column
# df['Preprocessed_Sentence'] = df['Sentence'].apply(preprocess_text)
#
# # Prepare the data
# if PREPROCESSING_FLAG == 1:
#     X = df['Preprocessed_Sentence']
# else:
#     X = df['Sentence']
#
# y = df['Sentiment']
#
# # Create a TF-IDF vectorizer
# tfidf_vectorizer = TfidfVectorizer()
# X_train_tfidf = tfidf_vectorizer.fit_transform(X)
#
# # Convert the labels to numerical values using LabelEncoder
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
#
# # Balancing the Dataset
# smote = SMOTE(sampling_strategy='auto', random_state=RAND_STATE)
# X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_encoded)
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_train_balanced, y_train_balanced, test_size=TEST_SIZE, random_state=RAND_STATE)
#
# # Convert X_train_tfidf to a dense numpy array
# X_train_tfidf_dense = X_train.toarray()
#
# # Convert X_test_tfidf to a dense numpy array
# X_test_tfidf_dense = X_test.toarray()
#
# #
# # if np.any(X_train_tfidf_dense < 0):
# #     # Handle or debug the presence of negative values in TF-IDF features
#     # print("Negative values found in TF-IDF features.")
#
# # X_train_tfidf_dense
# # y_encoded
#
# # In the following code, combining features extracted from TF-IDF and Word2Vec. This cell will overwrite variables from the previous cell. Only use when you want to use both techniques. Note: Accuracy falls in this case.
#
#
# # Constants
# PREPROCESSING_FLAG = 1
# SMOTE_FLAG = 1
# TEST_SIZE = 0.2
# RAND_STATE = 42
#
# # Apply data preprocessing to the 'Sentence' column and create 'Preprocessed_Sentence' column
# df['Preprocessed_Sentence'] = df['Sentence'].apply(preprocess_text)
#
# # Prepare the data
# if PREPROCESSING_FLAG == 1:
#     X = df['Preprocessed_Sentence']
# else:
#     X = df['Sentence']
#
# y = df['Sentiment']
#
# # Create a TF-IDF vectorizer
# tfidf_vectorizer = TfidfVectorizer()
# X_train_tfidf = tfidf_vectorizer.fit_transform(X)
#
# # Print the number of features added by TF-IDF
# # print(f"Number of features added by TF-IDF: {X_train_tfidf.shape[1]}")
#
# # Convert the labels to numerical values using LabelEncoder
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
#
# # Create Word2Vec features
# word2vec_features = []
#
# for doc in df['Preprocessed_Sentence']:
#     tokens = doc.split()
#     word2vec_embeddings = []
#
#     for token in tokens:
#         if token in loaded_model.wv:
#             word2vec_embeddings.append(loaded_model.wv[token])
#
#     # Calculate the average Word2Vec embedding for the document
#     avg_embedding = sum(word2vec_embeddings) / len(word2vec_embeddings) if word2vec_embeddings else [0.0] * loaded_model.vector_size
#
#     # Convert the average Word2Vec embedding to a dense array
#     avg_embedding_dense = [float(x) for x in avg_embedding]
#
#     word2vec_features.append(avg_embedding_dense)
#
# # Convert the list of Word2Vec features to a numpy array
# X_word2vec = np.array(word2vec_features)
#
# # Print the number of features added by Word2Vec
# # print(f"Number of features added by Word2Vec: {X_word2vec.shape[1]}")
#
# # Concatenate TF-IDF and Word2Vec features
# X_combined = np.concatenate((X_train_tfidf.toarray(), X_word2vec), axis=1)
#
# # Balancing the Dataset (if needed)
# if SMOTE_FLAG == 1:
#     smote = SMOTE(sampling_strategy='auto', random_state=RAND_STATE)
#     X_train_balanced, y_train_balanced = smote.fit_resample(X_combined, y_encoded)
# else:
#     X_train_balanced, y_train_balanced = X_combined, y_encoded
#
# # Split the data into training and testing sets
# X_train_tfidf_dense, X_test_tfidf_dense, y_train, y_test = train_test_split(X_train_balanced, y_train_balanced, test_size=TEST_SIZE, random_state=RAND_STATE)
#
# # Convert X_train_tfidf to a dense numpy array
# # X_train_tfidf_dense = X_train.toarray()
#
# # Convert X_test_tfidf to a dense numpy array
# # X_test_tfidf_dense = X_test.toarray()
#
# # Total number of features
# # print(f"Total Number of Features added by TF-IDF + Word2Vec= {X_train_tfidf.shape[1] + X_word2vec.shape[1]}")
# #
# # if np.any(X_train_tfidf_dense < 0):
# #     # Handle or debug the presence of negative values in TF-IDF features
# #     print("Negative values found in TF-IDF features.")
#
#
# individual_mode = False
#
# # Initialize classifiers as base models
# classifiers = {
#     #'Logistic Regression': LogisticRegression(max_iter=1000),
#     'SDG': SGDClassifier(),
#     # 'Random Forest': RandomForestClassifier(max_depth=10),
#     'Support Vector Machine': SVC(),
#     #'Naive Bayes': MultinomialNB(),
#     #  'Decision Tree': DecisionTreeClassifier(),
#     # 'Bagging': BaggingClassifier(),
#     # 'Gaussian Naive Bayes': GaussianNB(),
#     # 'Extreme Gradient Boosting (XGBoost)': XGBClassifier()
#
# }
#
# # Initialize blended_results_list
# results_list = []
# # Initialize blended_results_dict
# results_dict = {}
# # Initialize an empty list to store the rows of the results
# results_rows = []
# # Additional Information
# # Additional information about the code
# additional_info = {
#     'Total Classifiers': len(classifiers),
#     'Total Features': len(tfidf_vectorizer.get_feature_names_out()),
#     'Total Features': X_train_tfidf.shape[1] + X_word2vec.shape[1],
#     'Training Data Size': len(y_train),
#     'Test Data Size': len(X_test_tfidf_dense),
#     'Random State': RAND_STATE,  # Replace with the actual random_state value used in train_test_split
#     'Preprocessing': PREPROCESSING_FLAG,
#     'SMOTE': SMOTE_FLAG
# }
#
