import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def prepare_features(df, preprocess_text, loaded_model, preprocessing_flag=1, smote_flag=1, test_size=0.2,
                     rand_state=42):

    # Apply data preprocessing to the 'Sentence' column
    if preprocessing_flag == 1:
        df['Preprocessed_Sentence'] = df['Sentence'].apply(preprocess_text)
        X = df['Preprocessed_Sentence']
    else:
        X = df['Sentence']

    y = df['Sentiment']

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X)

    # Print the number of features added by TF-IDF
    print(f"Number of features added by TF-IDF: {X_train_tfidf.shape[1]}")

    # Convert the labels to numerical values using LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Create Word2Vec features
    word2vec_features = []

    for doc in df['Preprocessed_Sentence']:
        tokens = doc.split()
        word2vec_embeddings = []

        for token in tokens:
            if token in loaded_model.wv:
                word2vec_embeddings.append(loaded_model.wv[token])

        # Calculate the average Word2Vec embedding for the document
        avg_embedding = sum(word2vec_embeddings) / len(word2vec_embeddings) if word2vec_embeddings else [
                                                                                                            0.0] * loaded_model.vector_size

        # Convert the average Word2Vec embedding to a dense array
        avg_embedding_dense = [float(x) for x in avg_embedding]

        word2vec_features.append(avg_embedding_dense)

    # Convert the list of Word2Vec features to a numpy array
    X_word2vec = np.array(word2vec_features)

    # Print the number of features added by Word2Vec
    print(f"Number of features added by Word2Vec: {X_word2vec.shape[1]}")

    # Concatenate TF-IDF and Word2Vec features
    X_combined = np.concatenate((X_train_tfidf.toarray(), X_word2vec), axis=1)

    # Balancing the Dataset (if needed)
    if smote_flag == 1:
        smote = SMOTE(sampling_strategy='auto', random_state=rand_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_combined, y_encoded)
    else:
        X_train_balanced, y_train_balanced = X_combined, y_encoded

    # Split the data into training and testing sets
    X_train_tfidf_dense, X_test_tfidf_dense, y_train, y_test = train_test_split(X_train_balanced, y_train_balanced,
                                                                                test_size=test_size,
                                                                                random_state=rand_state)

    # Total number of features
    print(f"Total Number of Features added by TF-IDF + Word2Vec: {X_train_tfidf.shape[1] + X_word2vec.shape[1]}")

    # Optionally check if negative values are present in the TF-IDF features
    if np.any(X_train_tfidf_dense < 0):
        print("Negative values found in TF-IDF features.")

    # Return all the necessary components
    return X_train_tfidf_dense, X_test_tfidf_dense, y_train, y_test, tfidf_vectorizer, label_encoder
