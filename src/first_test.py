import string

import numpy as np
import torch
from nltk import word_tokenize
from collections import Counter
from torch.utils.data import Dataset, DataLoader

from src.model_prepare_dataset import load_pkl_data, prepare_data
from sklearn.model_selection import train_test_split


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


def prepare_embedding_matrix(word2index, embeddings_dictionary, embedding_dim=50):
    embedding_matrix = torch.zeros((len(word2index), embedding_dim))
    for word, index in word2index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = torch.from_numpy(embedding_vector)
    return embedding_matrix


class TextDataSet(Dataset):
    def __init__(self, texts, labels=None):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        if self.labels is not None:
            label = self.labels[index]
            return text, label
        else:
            return text


def texts2tensor(texts, word2index, pad_token=0, max_len=50):
    indexes_list = [[word2index.get(word, 0) for word in word_tokenize(text)] for text in texts]
    max_len = min(max_len, max([len(indexes) for indexes in indexes_list]))
    truncated_indexes = [indexes[:max_len] for indexes in indexes_list]
    padded_indexes = [indexes + [pad_token] * (max_len - len(indexes)) for indexes in truncated_indexes]
    return torch.LongTensor(padded_indexes)


def train_collate(batch_inputs, word2index):
    texts, labels = zip(*batch_inputs)
    input_tensor = texts2tensor(texts, word2index)
    return input_tensor, torch.LongTensor(labels)


def get_glove_matrix(train, test, glove_file_path):
    texts = train + test
    word2index, index2word, word_counter = build_vocab(texts)
    embeddings_dictionary = load_glove_embeddings(glove_file_path)
    embedding_matrix = prepare_embedding_matrix(word2index, embeddings_dictionary)

    train_dataset = TextDataSet(train, train)
    test_dataset = TextDataSet(test)

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True,
                              collate_fn=lambda batch: train_collate(batch, word2index))
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False,
                             collate_fn=lambda texts: texts2tensor(texts, word2index))

    return train_loader, test_loader, embedding_matrix


if __name__ == "__main__":
    movie_conversations = load_pkl_data(r"C:\Users\Marco\dev\git\DL-project\data\movie_conversations.pkl")

    all_pairs = prepare_data(movie_conversations)

    train_pairs, test_pairs = train_test_split(all_pairs, shuffle=True, test_size=0.3, random_state=42)

    glove_file_path = './glove.6B.50d.txt'

    train_loader, test_loader, embedding_matrix = main(train_pairs, test_pairs, glove_file_path)
