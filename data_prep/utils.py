import pandas as pd
from collections import defaultdict
from tqdm import tqdm


def count_words(text):
    """Count the number of occurrences of each word in a set of text"""
    count_dict = defaultdict(int)
    for sentence in text:
        for word in sentence.split():
            count_dict[word] += 1
    return dict(count_dict)


def convert_to_ints(text, word_count, unk_count, vocab_to_int, eos=False):
    """Convert words in text to an integer.
       If word is not in vocab_to_int, use UNK's integer.
       Total the number of words and UNKs.
       Add EOS token to the end of texts"""
    ints = []
    for sentence in text:
        sentence_ints = []
        for word in sentence.split():
            word_count += 1
            if word in vocab_to_int:
                sentence_ints.append(vocab_to_int[word])
            else:
                sentence_ints.append(vocab_to_int["<UNK>"])
                unk_count += 1
        if eos:
            sentence_ints.append(vocab_to_int["<EOS>"])
        ints.append(sentence_ints)
    return ints, word_count, unk_count


def convert_vocab_to_int(word_counts, embeddings_index, threshold=20):
    # Limit the vocab that we will use to words that appear ≥ threshold or are in GloVe

    # dictionary to convert words to integers
    vocab_to_int = {}

    value = 0
    for word, count in word_counts.items():
        if count >= threshold or word in embeddings_index:
            vocab_to_int[word] = value
            value += 1

    # Special tokens that will be added to our vocab
    codes = ["<UNK>", "<PAD>", "<EOS>", "<GO>"]

    # Add codes to vocab
    for code in codes:
        vocab_to_int[code] = len(vocab_to_int)

    # Dictionary to convert integers to words
    int_to_vocab = {}
    for word, value in vocab_to_int.items():
        int_to_vocab[value] = word

    usage_ratio = round(len(vocab_to_int) / len(word_counts), 4) * 100

    print("Total number of unique words:", len(word_counts))
    print("Number of words we will use:", len(vocab_to_int))
    print("Percent of words we will use: {}%".format(usage_ratio))


def unk_counter(sentence, vocab_to_int):
    """Counts the number of time UNK appears in a sentence."""
    unk_count = 0
    for word in sentence:
        if word == vocab_to_int["<UNK>"]:
            unk_count += 1
    return unk_count


def clean_reviews(lengths_texts, int_texts, vocab_to_int):
    # Sort the summaries and texts by the length of the texts, shortest to longest
    # Limit the length of summaries and texts based on the min and max ranges.
    # Remove reviews that include too many UNKs

    # sorted_summaries = []
    sorted_texts = []
    max_text_length = 150
    # max_summary_length = 13
    min_length = 2
    unk_text_limit = 10
    # unk_summary_limit = 0

    for _ in tqdm(range(min(lengths_texts.counts), max_text_length)):
        for count in range(len(int_texts)):
            if (unk_counter(int_texts[count], vocab_to_int) <= unk_text_limit and len(
                    int_texts[count]) >= min_length and len(int_texts[count]) <= max_text_length):
                sorted_texts.append(int_texts[count])

    # Compare lengths to ensure they match
    print(len(sorted_texts))


def create_lengths(text):
    """Create a data frame of the sentence lengths from a text"""
    lengths = []
    for sentence in tqdm(text):
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])
