import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def clean_dataset(input_path, output_path):
    data = pd.read_csv(input_path)
    data['cleaned_text'] = data['text'].apply(clean_text)
    data.to_csv(output_path, index=False)

if __name__ == "__main__":
    clean_dataset('data/quora_question_pairs.csv', 'data/cleaned_quora_question_pairs.csv')
