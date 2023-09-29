import string
#defining the function to remove punctuation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
nltk.download('stopwords')

def remove_whitespace(text):
    return text.strip()

def lowering(text):
    return text.lower()

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_punctuation(text):
    return "".join([i for i in text if i not in string.punctuation])

def remove_stopword(text):
    tokens = word_tokenize(text)
    english_stopwords = stopwords.words('english')
    tokens_wo_stopwords = [t for t in tokens if t not in english_stopwords]
    return " ".join(tokens_wo_stopwords)

def stem_words(text):
    word_tokens = word_tokenize(text)
    stems = [stemmer.stem(word) for word in word_tokens]
    return " ".join(stems)
 
def text_preprocessing(text):
    text_1 = remove_whitespace(text)
    text_2 = lowering(text_1)
    text_3 = remove_numbers(text_2)
    text_4 = remove_punctuation(text_3)
    text_5 = remove_stopword(text_4)
    return text_5