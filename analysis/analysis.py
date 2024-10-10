# analysis.py

import os
import psutil
import platform
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from itertools import combinations
from multiprocessing import Process, Queue
from sklearn.ensemble import VotingClassifier, BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, cohen_kappa_score
from imblearn.over_sampling import SMOTE
from textblob import classifiers
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


# ======================== Helper Methods ==========================

# Convert bytes to MB or GB
def bytes_to_mb_or_gb(byte_value):
    if byte_value >= 1024 ** 3:  # GB
        return f"{byte_value / (1024 ** 3):.2f} GB"
    elif byte_value >= 1024 ** 2:  # MB
        return f"{byte_value / (1024 ** 2):.2f} MB"
    else:
        return f"{byte_value} bytes"


# ==================== Blended Classifier Function ===================

def train_and_evaluate_blended_classifier(classifier_combo, additional_info, num_classifiers):
    print(f"Blending the following classifiers: {classifier_combo}")
    start_time = time.time()  # Record the start time

    num_cores = multiprocessing.cpu_count()
    processor_type = platform.processor()
    os_name = platform.system() + '(' + platform.release() + ')'
    ram_total = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    ram_total_mb = ram_total / (1024 ** 2)

    cpu_before = psutil.cpu_percent()
    available_memory_before = psutil.virtual_memory().available

    # Create the voting classifier with the selected combination
    selected_classifiers = [(name, classifiers[name]) for name in classifier_combo]
    voting_classifier = VotingClassifier(estimators=selected_classifiers, voting='hard')

    # Train the voting classifier
    voting_classifier.fit(X_train_tfidf_dense, y_train)

    # Predictions and evaluation
    final_predictions = voting_classifier.predict(X_test_tfidf_dense)
    accuracy = accuracy_score(y_test, final_predictions)
    precision = precision_score(y_test, final_predictions, average='weighted', zero_division=1)
    recall = recall_score(y_test, final_predictions, average='weighted')
    f1 = f1_score(y_test, final_predictions, average='weighted')
    cm = confusion_matrix(y_test, final_predictions)
    kappa = cohen_kappa_score(y_test, final_predictions)

    end_time = time.time()
    execution_time = end_time - start_time

    cpu_after = psutil.cpu_percent()
    available_memory_after = psutil.virtual_memory().available
    cpu_usage = max(cpu_after - cpu_before, 0)
    memory_usage = max(available_memory_after - available_memory_before, 0)
    memory_usage_str = bytes_to_mb_or_gb(memory_usage)

    # Print CPU and Memory usage
    print(f"CPU Usage: {cpu_usage:.2f}%")
    print(f"Memory Usage: {memory_usage_str}")

    # Store results
    results_dict = {
        'Blended Classifiers': classifier_combo,
        'Accuracy': accuracy,
        'Kappa': kappa,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Confusion Matrix': cm,
        'Execution Time (s)': execution_time,
        'Total Classifiers': additional_info['Total Classifiers'],
        'Total Features': additional_info['Total Features'],
        'Training Data Size': additional_info['Training Data Size'],
        'Test Data Size': additional_info['Test Data Size'],
        'Random State': additional_info['Random State'],
        'Preprocessing': additional_info['Preprocessing'],
        'SMOTE': additional_info['SMOTE'],
        'Total CPU Cores': num_cores,
        'CPU Usage (%)': cpu_usage,
        'Total RAM': ram_total_mb,
        'Memory Usage': memory_usage_str,
        'Processor Type': processor_type,
        'OS': os_name
    }

    print(results_dict)
    return results_dict


# ================== Word2Vec Model Generation =====================

def generate_word2vec_model(df):
    df['Preprocessed_Sentence'] = df['Sentence'].apply(preprocess_text)
    corpus = df['Preprocessed_Sentence'].apply(lambda x: x.split())

    word2vec_model = Word2Vec(sentences=corpus, vector_size=300, window=5, min_count=1, sg=0, epochs=10)

    latest_loss = word2vec_model.get_latest_training_loss()
    print(f"Latest Training Loss: {latest_loss}")

    vocabulary_words = word2vec_model.wv.index_to_key
    print("First 10 words in the vocabulary:")
    print(vocabulary_words[:10])

    word2vec_model.save("word2vec_model.bin")
    loaded_model = Word2Vec.load("word2vec_model.bin")

    return loaded_model


# ==================== Preprocessing and Splitting ===================

def preprocess_and_split_data(df, preprocess_flag=1, smote_flag=1, test_size=0.2, rand_state=42):
    df['Preprocessed_Sentence'] = df['Sentence'].apply(preprocess_text)
    X = df['Preprocessed_Sentence'] if preprocess_flag == 1 else df['Sentence']
    y = df['Sentiment']

    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    if smote_flag == 1:
        smote = SMOTE(sampling_strategy='auto', random_state=rand_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_encoded)
    else:
        X_train_balanced, y_train_balanced = X_train_tfidf, y_encoded

    X_train, X_test, y_train, y_test = train_test_split(X_train_balanced, y_train_balanced, test_size=test_size,
                                                        random_state=rand_state)

    X_train_tfidf_dense = X_train.toarray()
    X_test_tfidf_dense = X_test.toarray()

    return X_train_tfidf_dense, X_test_tfidf_dense, y_train, y_test, label_encoder


# ==================== Prepare Data for Word2Vec =====================

def prepare_data(df, loaded_model, preprocessing_flag=1, smote_flag=1, test_size=0.2, rand_state=42):
    df['Preprocessed_Sentence'] = df['Sentence'].apply(preprocess_text)
    X = df['Preprocessed_Sentence'] if preprocessing_flag == 1 else df['Sentence']
    y = df['Sentiment']

    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    word2vec_features = []
    for doc in df['Preprocessed_Sentence']:
        tokens = doc.split()
        word2vec_embeddings = [loaded_model.wv[token] for token in tokens if token in loaded_model.wv]
        avg_embedding = sum(word2vec_embeddings) / len(word2vec_embeddings) if word2vec_embeddings else [
                                                                                                            0.0] * loaded_model.vector_size
        word2vec_features.append(avg_embedding)

    X_word2vec = np.array(word2vec_features)
    X_combined = np.concatenate((X_train_tfidf.toarray(), X_word2vec), axis=1)

    if smote_flag == 1:
        smote = SMOTE(sampling_strategy='auto', random_state=rand_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_combined, y_encoded)
    else:
        X_train_balanced, y_train_balanced = X_combined, y_encoded

    X_train_dense, X_test_dense, y_train, y_test = train_test_split(X_train_balanced, y_train_balanced,
                                                                    test_size=test_size, random_state=rand_state)

    return X_train_dense, X_test_dense, y_train, y_test, label_encoder


# ==================== Initialize Classifiers =======================

def initialize_classifiers():
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'SGD': SGDClassifier(),
        'Support Vector Machine': SVC(),
        'Naive Bayes': MultinomialNB(),
        'Bagging': BaggingClassifier(),
        # Add more classifiers as needed
    }
    return classifiers


# ================= Additional Information ==========================

additional_info = {
    'Total Classifiers': len(classifiers),
    'Total Features': len(tfidf_vectorizer.get_feature_names_out()),
    # 'Total Features': X_train_tfidf.shape[1] + X_word2vec.shape[1],
    'Training Data Size': len(y_train),
    'Test Data Size': len(X_test_tfidf_dense),
    'Random State': RAND_STATE,  # Replace with the actual random_state value used in train_test_split
    'Preprocessing': PREPROCESSING_FLAG,
    'SMOTE': SMOTE_FLAG
}

