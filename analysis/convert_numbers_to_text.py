import re  # Make sure this import is present
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from num2words import num2words  # Ensure you have this import for num2words conversion

# Custom function to convert numbers to text if needed
def convert_numbers_to_text(tokens):
    # Custom financial domain-specific stopwords
    custom_stopwords = set(["stock", "price", "earnings", "report", "investors", "company"])

    converted_tokens = []
    for token in tokens:
        # Check if it's a number (integer or decimal)
        if re.match(r'^-?\d+(?:\.\d+)?$', token):
            converted_token = num2words(token, to='currency')  # Convert number to text with currency format
            converted_tokens.extend(converted_token.split())  # Split the converted text
        else:
            converted_tokens.append(token)
    return converted_tokens  # Make sure to return the correct result
