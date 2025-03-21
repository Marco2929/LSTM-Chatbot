import pickle
import re
import random
from typing import List, Tuple

import numpy as np
from datasets import load_dataset
import unicodedata
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Vocab:
    def __init__(self):
        self.trimmed = False
        self.word_to_index = {}
        self.word_to_count = {}
        self.index_to_word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_of_words = 3  # Count SOS, EOS, PAD

    def add_sentence(self, sentence: str):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word: str):
        if word not in self.word_to_index:
            self.word_to_index[word] = self.num_of_words
            self.word_to_count[word] = 1
            self.index_to_word[self.num_of_words] = word
            self.num_of_words += 1
        else:
            self.word_to_count[word] += 1

    def trim(self, min_count: int):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = [word for word, count in self.word_to_count.items() if count >= min_count]

        # Reinitialize dictionaries
        self.word_to_index = {}
        self.word_to_count = {}
        self.index_to_word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_of_words = 3  # Count default tokens

        for word in keep_words:
            self.add_word(word)


def load_pkl_data(pkl_file_path: str) -> List[List[str]]:
    with open(pkl_file_path, 'rb') as pkl_file:
        return pickle.load(pkl_file)


def load_huggingface_dataset() -> List[str]:
    dataset = load_dataset("daily_dialog", "default", trust_remote_code=True)
    hf_dataset_train = dataset.data['train']['dialog'].to_pylist()
    hf_dataset_test = dataset.data['test']['dialog'].to_pylist()
    hf_dataset_validation = dataset.data['validation']['dialog'].to_pylist()

    return hf_dataset_train + hf_dataset_test + hf_dataset_validation


def unicode_to_ascii(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def normalize_string(s: str) -> str:
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"[.!?]", "", s)  # Remove all punctuation marks
    s = re.sub(r"[^a-zA-Z]+", " ", s)  # Remove non-alphabetic characters
    return s.strip()


def prepare_data(conversations: List[List[str]]) -> List[List[str]]:
    return [[normalize_string(conversation[i]), normalize_string(conversation[i + 1])]
            for conversation in conversations for i in range(len(conversation) - 1)]


def filter_pairs(pairs: List[List[str]], max_length: int) -> List[List[str]]:
    return [pair for pair in pairs if len(pair[0].split()) < max_length and len(pair[1].split()) < max_length]


def trim_rare_words(vocab: Vocab, pairs: List[List[str]], min_count: int) -> List[List[str]]:
    vocab.trim(min_count)
    return [pair for pair in pairs
            if all(word in vocab.word_to_index for word in pair[0].split()) and
            all(word in vocab.word_to_index for word in pair[1].split())]


def filter_and_trim_data(vocab: Vocab, pairs: List[List[str]], max_length: int) -> Tuple[List[List[str]], Vocab]:
    pairs = filter_pairs(pairs, max_length)
    for pair in pairs:
        vocab.add_sentence(pair[0])
        vocab.add_sentence(pair[1])
    pairs = trim_rare_words(vocab, pairs, min_count=3)
    return pairs, vocab


def prepare_dataset(max_length: int) -> Tuple[Vocab, List[List[str]], List[List[str]]]:
    print("Load conversations...")
    movie_conversations = load_pkl_data(r"C:\Users\Marco\dev\git\DL-project\data\movie_conversations.pkl")
    daily_conversations = load_huggingface_dataset()

    all_conversations = movie_conversations + daily_conversations
    print(f"Number of raw dialogs: {len(all_conversations)}")
    all_pairs = prepare_data(all_conversations)
    print(f"Number of prepared sentence pairs: {len(all_pairs)}")
    vocab = Vocab()

    all_pairs, vocab = filter_and_trim_data(vocab, all_pairs, max_length)
    print(f"Number of sentence pairs after filtering and trimming: {len(all_pairs)}")
    train_pairs, test_pairs = train_test_split(all_pairs, shuffle=True, test_size=0.2, random_state=42)

    return vocab, train_pairs, test_pairs


def convert_word_to_index(vocab: Vocab, sentence: str):
    return [vocab.word_to_index[word] for word in sentence.split(' ')]


def process_pairs(pairs, vocab, max_length):
    num_pairs = len(pairs)
    input_sequences = np.zeros((num_pairs, max_length), dtype=np.int32)
    target_sequences = np.zeros((num_pairs, max_length), dtype=np.int32)

    for idx, (input_sentence, target_sentence) in enumerate(pairs):
        input_indices = convert_word_to_index(vocab, input_sentence)
        target_indices = convert_word_to_index(vocab, target_sentence)
        input_indices.append(EOS_token)
        target_indices.append(EOS_token)
        input_sequences[idx, :len(input_indices)] = input_indices
        target_sequences[idx, :len(target_indices)] = target_indices

    return input_sequences, target_sequences


def get_dataloader(vocab, batch_size, max_length, train_pairs, test_pairs):
    train_input_sequences, train_target_sequences = process_pairs(train_pairs, vocab, max_length)
    test_input_sequences, test_target_sequences = process_pairs(test_pairs, vocab, max_length)

    train_tensor_input = torch.LongTensor(train_input_sequences).to(device)
    train_tensor_target = torch.LongTensor(train_target_sequences).to(device)
    test_tensor_input = torch.LongTensor(test_input_sequences).to(device)
    test_tensor_target = torch.LongTensor(test_target_sequences).to(device)

    train_dataset = TensorDataset(train_tensor_input, train_tensor_target)
    test_dataset = TensorDataset(test_tensor_input, test_tensor_target)

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

    return train_dataloader, test_dataloader
