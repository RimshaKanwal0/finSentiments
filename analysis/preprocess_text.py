import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from .convert_numbers_to_text import convert_numbers_to_text


# Tokenization and preprocessing function
def preprocess_text(text):
    # Custom financial domain-specific stopwords
    custom_stopwords = set(["stock", "price", "earnings", "report", "investors", "company"])

    # Convert text to lowercase
    text = text.lower()

    # Remove HTML tags and URLs
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Tokenize text using NLTK's word_tokenize
    tokens = word_tokenize(text)

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Apply lemmatization, filter out stopwords, and custom financial stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens if len(word) > 1 and word.lower() not in stopwords.words('english') and word.lower() not in custom_stopwords]

    # Filter out single characters and symbols
    tokens = [word for word in tokens if len(word) > 1]  # or re.match(r'[$€%]', word)

    # Convert numbers to text representations (including currency format)
    tokens = convert_numbers_to_text(tokens)

    # Join the tokens back to form preprocessed text
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text
