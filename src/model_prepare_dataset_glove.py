import pickle
import re
import random
from typing import List, Tuple

import numpy as np
from datasets import load_dataset
import unicodedata
import torch
import string

from nltk import word_tokenize
from collections import Counter

from src.model_prepare_dataset import load_pkl_data, prepare_data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


def remove_punct(text):
    table = str.maketrans('', '', string.punctuation)
    return text.translate(table)


def build_vocab(texts):
    word_counter = Counter()
    for text in texts:
        for words in text:
            words_tokenized = list(word_tokenize(remove_punct(words)))
            word_counter.update(words_tokenized)
    print("word num:", len(word_counter))

    word2index = {'unk': 0}
    for i, word in enumerate(word_counter.keys()):
        word2index[word] = i + 1
    index2word = {v: k for k, v in word2index.items()}
    return word2index, index2word, word_counter


def load_glove_embeddings(file_path):
    embeddings_dictionary = {}
    with open(file_path, encoding="utf8") as glove_file:
        for line in glove_file:
            records = line.split()
            word = records[0]
            vector_dimensions = np.asarray(records[1:], dtype='float32')
            embeddings_dictionary[word] = vector_dimensions
    return embeddings_dictionary


def load_pkl_data(pkl_file_path: str) -> List[List[str]]:
    with open(pkl_file_path, 'rb') as pkl_file:
        return pickle.load(pkl_file)


def load_huggingface_dataset() -> List[str]:
    dataset = load_dataset("daily_dialog", "default", trust_remote_code=True)
    return dataset.data['train']['dialog'].to_pylist()


def unicode_to_ascii(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def normalize_string(s: str) -> str:
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s.strip()


def prepare_data(conversations: List[List[str]]) -> List[List[str]]:
    return [[normalize_string(conversation[i]), normalize_string(conversation[i + 1])]
            for conversation in conversations for i in range(len(conversation) - 1)]


def prepare_embedding_matrix(word2index, embeddings_dictionary, embedding_dim=50):
    embedding_matrix = torch.zeros((len(word2index), embedding_dim))
    for word, index in word2index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = torch.from_numpy(embedding_vector)
    return embedding_matrix


def get_glove_matrix(train, test):
    glove_file_path = "./glove.6B.50d.txt"
    texts = train + test
    word2index, index2word, word_counter = build_vocab(texts)
    embeddings_dictionary = load_glove_embeddings(glove_file_path)
    embedding_matrix = prepare_embedding_matrix(word2index, embeddings_dictionary)

    return embedding_matrix, word2index


def prepare_dataset():
    print("Load conversations...")
    movie_conversations = load_pkl_data(r"C:\Users\Marco\dev\git\DL-project\data\movie_conversations.pkl")

    all_pairs = prepare_data(movie_conversations)

    train_pairs, test_pairs = train_test_split(all_pairs, shuffle=True, test_size=0.3, random_state=42)

    embeddings_matrix, word2index = get_glove_matrix(train_pairs, test_pairs)

    return train_pairs, test_pairs, embeddings_matrix, word2index


def texts2tensor(texts, word2index, max_len=15, pad_token=PAD_token):
    indexes_list = [[word2index.get(word, 0) for word in word_tokenize(text)] for text in texts]
    max_len = min(max_len, max([len(indexes) for indexes in indexes_list]))
    truncated_indexes = [indexes[:max_len] for indexes in indexes_list]
    padded_indexes = [indexes + [pad_token] * (max_len - len(indexes)) for indexes in truncated_indexes]
    return torch.LongTensor(padded_indexes)


def train_collate(batch_inputs):
    texts, labels = zip(*batch_inputs)
    input_tensor = texts2tensor(texts, word2index)
    return input_tensor, torch.LongTensor(labels)


if __name__ == '__main__':
    train_pairs, test_pairs, embeddings_matrix, word2index = prepare_dataset()

    train_loader = DataLoader(train_pairs, batch_size=64, shuffle=True, collate_fn=train_collate)
    test_loader = DataLoader(test_pairs, batch_size=64, shuffle=False,
                             collate_fn=lambda texts: texts2tensor(texts, word2index))

    pass
