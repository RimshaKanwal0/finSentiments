import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from .preprocess_text import preprocess_text  # Assuming preprocessing is from another file

# Constants (can be modified as per requirement)
PREPROCESSING_FLAG = 1
SMOTE_FLAG = 1
TEST_SIZE = 0.2
RAND_STATE = 42


def vectorize_balance_split_data(df, preprocessing_flag=PREPROCESSING_FLAG, smote_flag=SMOTE_FLAG, test_size=TEST_SIZE,
                                 rand_state=RAND_STATE):

    # Apply preprocessing to the 'Sentence' column
    if preprocessing_flag == 1:
        df['Preprocessed_Sentence'] = df['Sentence'].apply(preprocess_text)
        X = df['Preprocessed_Sentence']
    else:
        X = df['Sentence']

    # Prepare the labels
    y = df['Sentiment']

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X)

    # Label Encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Balancing the Dataset with SMOTE (if smote_flag is set to 1)
    if smote_flag == 1:
        smote = SMOTE(sampling_strategy='auto', random_state=rand_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_tfidf, y_encoded)
    else:
        X_train_balanced, y_train_balanced = X_train_tfidf, y_encoded

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_train_balanced, y_train_balanced, test_size=test_size,
                                                        random_state=rand_state)

    # Convert X_train and X_test to dense numpy arrays
    X_train_tfidf_dense = X_train.toarray()
    X_test_tfidf_dense = X_test.toarray()

    # Optionally check if negative values are present in the TF-IDF features
    if np.any(X_train_tfidf_dense < 0):
        print("Negative values found in TF-IDF features.")

    # Return all the necessary components
    return X_train_tfidf_dense, X_test_tfidf_dense, y_train, y_test, tfidf_vectorizer, label_encoder
