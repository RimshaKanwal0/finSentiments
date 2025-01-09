import pandas as pd
from gensim.models import Word2Vec
from .preprocess_text import preprocess_text  # Import your custom preprocessing function


def train_word2vec_model(dataframe):
    """
    Trains a Word2Vec model using the provided dataframe containing text sentences.
    Returns the trained model, the latest training loss, and some vocabulary information.
    """
    # Preprocess the sentences in the dataframe using the external preprocess_text function
    dataframe['Preprocessed_Sentence'] = dataframe['Sentence'].apply(preprocess_text)

    # Tokenize sentences into words (split by space)
    corpus = dataframe['Preprocessed_Sentence'].apply(lambda x: x.split())

    # Train the Word2Vec model
    word2vec_model = Word2Vec(sentences=corpus, vector_size=300, window=5, min_count=1, sg=0, epochs=10)

    # Monitor Loss: Track the loss (negative log likelihood) during training
    latest_loss = word2vec_model.get_latest_training_loss()

    # Get the first 10 words in the vocabulary
    vocabulary_words = word2vec_model.wv.index_to_key
    first_10_words = vocabulary_words[:10]

    # Optionally, check word similarity (for example: "stock" and "market")
    similarity_score = word2vec_model.wv.similarity('stock', 'market')

    # Save the trained model
    word2vec_model.save("word2vec_model.bin")

    return word2vec_model, latest_loss, first_10_words, similarity_score
