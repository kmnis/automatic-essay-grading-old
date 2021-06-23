import re
from nltk.corpus import stopwords
import numpy as np
import pickle
import os
from tqdm import tqdm
import pandas as pd
# from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence

pj = os.path.join

# A list of contractions from https://stackoverflow.com/q/19790188/9865225
contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
}


def clean_text(text, remove_stopwords=True, remove_contractions=True):
    """Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings"""

    # Convert words to lower case
    text = text.lower()

    # Replace contractions with their longer forms
    if remove_contractions:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)

    # Format words and remove unwanted characters
    text = re.sub(r'https?://.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'<a href', ' ', text)
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)

    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if w not in stops]
        text = " ".join(text)

    return text


def load_embeddings(glove_dir, words):
    embeddings = {}
    with open(glove_dir, encoding='utf-8') as f:
        for line in tqdm(f, total=400000):
            values = line.split(' ')
            word = values[0].lower()
            if len({word} & words):
                embedding = np.asarray(values[1:], dtype='float32')
                embeddings[word] = embedding

    embeddings["<PAD>"] = 0
    embeddings["<UNK>"] = 0
    print('Total number of word embeddings:', len(embeddings))
    return embeddings


def create_score(df):
    df.loc[:, 'score'] = 0

    # For essay_set = 1
    ind = df.index[df['essay_set'] == 1]
    for i in ind:
        df.loc[i, 'score'] = df['domain1_score'][i] / 12

    # For essay_set = 2
    ind = df.index[df['essay_set'] == 2]
    for i in ind:
        df.loc[i, 'score'] = ((df['domain1_score'][i] / 6) + (df['domain2_score'][i] / 4)) / 2

    # For essay_set = 3
    ind = df.index[df['essay_set'] == 3]
    for i in ind:
        df.loc[i, 'score'] = df['domain1_score'][i] / 3

    # For essay_set = 4
    ind = df.index[df['essay_set'] == 4]
    for i in ind:
        df.loc[i, 'score'] = df['domain1_score'][i] / 3

    # For essay_set = 5
    ind = df.index[df['essay_set'] == 5]
    for i in ind:
        df.loc[i, 'score'] = df['domain1_score'][i] / 4

    # For essay_set = 6
    ind = df.index[df['essay_set'] == 6]
    for i in ind:
        df.loc[i, 'score'] = df['domain1_score'][i] / 4

    # For essay_set = 7
    ind = df.index[df['essay_set'] == 7]
    for i in ind:
        df.loc[i, 'score'] = df['domain1_score'][i] / 30

    # For essay_set = 8
    ind = df.index[df['essay_set'] == 8]
    for i in ind:
        df.loc[i, 'score'] = df['domain1_score'][i] / 60

    df = df[["essay", "score"]]
    return df


def process_essay_df(df, glove_dir, embedding_dir):
    words = []
    for i in tqdm(range(len(df)), total=len(df)):
        cleaned_essay = clean_text(df.essay[i])
        word_sequence = text_to_word_sequence(cleaned_essay)
        words.extend(word_sequence)

    print("Total number of words:", len(words))
    print("Total number of unique words:", len(set(words)))
    # tokenizer = Tokenizer(num_words=5000, oov_token='<UNK>')
    # tokenizer.fit_on_texts(words)
    # words = list(tokenizer.word_index.keys())
    embeddings = load_embeddings(glove_dir, set(words))

    with open(embedding_dir, "wb") as f:
        pickle.dump(embeddings, f)


if __name__ == '__main__':
    data_dir = pj(os.path.abspath(__file__).strip("preprocessing.py"), "../data")
    df = pd.read_csv(pj(data_dir, "raw/training_set_rel3.tsv"), sep='\t', encoding="ISO-8859-1")
    process_essay_df(df, pj(data_dir, "glove/glove.6B.50d.txt"), pj(data_dir, "processed/embeddings.p"))

    print("Getting score for each essay")
    df = create_score(df)
    df.to_csv(pj(data_dir, "processed/data.csv"), index=False)
