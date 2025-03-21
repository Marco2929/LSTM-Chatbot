import os
import pickle
import re

import numpy as np
import seaborn as sns
import torch
from evaluate import load
from matplotlib import pyplot as plt

from src.model_prepare_dataset import prepare_dataset, Vocab, normalize_string, convert_word_to_index
from src.model_train import EncoderLSTM, DecoderLSTMAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EOS_token = 2  # End-of-sentence token

questions = [
    'What is your name?',
    'Where do you live?',
    'What is your favorite color?',
    'What do you do for a living?',
    'What is your hobby?',
    'Do you have any pets?',
    'What is your favorite book?',
    'What is your favorite movie?',
    'What is your favorite song?',
    'Where were you born?',
    'What is your favorite sport?',
    'What languages do you speak?',
    'What is your favorite season?',
    'What is your favorite holiday?',
    'What is your favorite drink?',
    'What is your favorite TV show?',
    'What is your favorite animal?',
    'What is your favorite subject in school?',
    'What is your favorite game?',
    'What is your dream job?'
    "Are we friends?"
]


def tensor_from_sentence(vocab: Vocab, sentence: str):
    indexes = convert_word_to_index(vocab, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)


def inference(encoder: EncoderLSTM, decoder: DecoderLSTMAttention, sentence: str, vocab: Vocab):
    with torch.no_grad():
        input_tensor = tensor_from_sentence(vocab, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            decoded_words.append(vocab.index_to_word[idx.item()])
    return decoded_words, decoder_attn


def format_sentence(sentence):
    if not sentence:
        return sentence
    sentence = sentence.strip()

    # Dictionary of contractions to replace
    contractions = {
        r'\b s \b': " is ",
        r'\b re \b': " are ",
    }

    # Perform the substitutions
    for pattern, replacement in contractions.items():
        sentence = re.sub(pattern, replacement, sentence)

    if not sentence.endswith('.'):
        sentence += '.'

    return sentence[0].upper() + sentence[1:]


def dataset_input(encoder: EncoderLSTM, decoder: DecoderLSTMAttention, vocab: Vocab):
    # ANSI escape codes for colors and bold text
    yellow_bold = "\033[1;33m"
    green = "\033[0;32m"
    red = "\033[0;31m"
    cyan = "\033[0;36m"
    reset = "\033[0m"

    perplexity = load("perplexity", module_type="metric")

    perplexities = []

    # Evaluating each question
    for question in questions:
        try:
            # Normalize question
            normalized_question = normalize_string(question)
            # Evaluate question
            output_words, _ = inference(encoder, decoder, normalized_question, vocab)
            # Format and print response sentence
            output_words = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            output_sentence = ' '.join(output_words)
            print(f"{yellow_bold}Question:{reset} {cyan}{question}{reset}")
            print(f"{green}Answer:{reset} {output_sentence}")
            results = perplexity.compute(model_id='gpt2',
                                         add_start_token=True,
                                         predictions=[output_sentence])
            print(f"{green}Perplexity score:{reset} {results['mean_perplexity']}")
            perplexities.append(results['mean_perplexity'])
            print()

        except KeyError:
            print(
                f"{red}Some of the input words in the question are unknown, please formulate the sentence differently.{reset}")

    print(f"{yellow_bold}Overall perplexity score:{reset} {np.mean(perplexities)}")


def user_input(encoder: EncoderLSTM, decoder: DecoderLSTMAttention, vocab: Vocab):
    # ANSI escape codes for colors and bold text
    yellow_bold = "\033[1;33m"
    green = "\033[0;32m"
    red = "\033[0;31m"
    cyan = "\033[0;36m"
    reset = "\033[0m"

    # Greeting message
    print(f"{yellow_bold}Hello! How can I assist you today?{reset}")
    while True:
        try:
            # Get input sentence
            input_sentence = input(f'{cyan}> {reset}')
            # Check if it is quit case
            if input_sentence.lower() in ['q', 'quit']:
                print(f"{yellow_bold}Goodbye! Have a great day!{reset}")
                break
            # Normalize sentence
            input_sentence = normalize_string(input_sentence)
            # Evaluate sentence
            output_words, _ = inference(encoder, decoder, input_sentence, vocab)
            # Format and print response sentence
            output_words = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            formatted_response = format_sentence(' '.join(output_words))
            print(f"{green}Answer:{reset} {formatted_response}")

        except KeyError:
            print(f"{red}Some of the input you used is unknown, please formulate the sentence differently.{reset}")


def load_checkpoint(path: str, encoder: EncoderLSTM, decoder: DecoderLSTMAttention):
    if os.path.isfile(path):
        print(f"Loading checkpoint '{path}'")
        checkpoint = torch.load(path)
        epoch = checkpoint['epoch']
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print(f"Loaded checkpoint '{path}' (epoch {epoch})")
        return encoder, decoder
    else:
        print(f"No checkpoint found at '{path}'")
        return None, None


def load_pkl_vocab(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        print(f"An error occurred while loading the pickle file: {e}")
        return None


def visualize_attention_weights(input_sentence, words, attention_weights):

    # Move attention weights to CPU and convert to numpy
    attention_weights_cpu = attention_weights.cpu().detach().numpy()

    # Filter out 'PAD' and 'EOS' tokens from the response and get their indices
    filtered_response_tokens = []
    filtered_indices = []

    for idx, token in enumerate(words):
        if token not in ['PAD', 'EOS']:
            filtered_response_tokens.append(token)
            filtered_indices.append(idx)

    # Filter the attention weights to exclude 'PAD' and 'EOS' tokens in both rows and columns
    filtered_attention_weights = attention_weights_cpu[0, filtered_indices, :]
    filtered_attention_weights = filtered_attention_weights[:, filtered_indices]

    plot_attention_weights(input_sentence, filtered_response_tokens, filtered_attention_weights)


def plot_attention_weights(input_sentence, response_tokens, attention_weights):
    fig, ax = plt.subplots()
    cax = ax.matshow(attention_weights, cmap='viridis')

    ax.set_xticklabels([''] + input_sentence.split(), rotation=90)
    ax.set_yticklabels([''] + response_tokens)

    plt.title('Bahdanau Attention Weights')
    plt.colorbar(cax)
    plt.show()


if __name__ == '__main__':
    # Configure training/optimization
    hidden_size = 500
    encoder_layer_number = 2
    decoder_layer_number = 2
    batch_size = 256
    max_length = 10
    dropout = 0.1

    current_path = r"C:\Users\Marco\dev\git\chatbot_notebook"

    name = "model"

    vocab: Vocab = load_pkl_vocab(os.path.join(current_path, f"model_checkpoint/{name}.pkl"))

    encoder = EncoderLSTM(input_size=vocab.num_of_words, num_layers=encoder_layer_number, hidden_size=hidden_size).to(
        device)
    decoder = DecoderLSTMAttention(hidden_size=hidden_size, output_size=vocab.num_of_words,
                                   num_layers=decoder_layer_number,
                                   max_length=max_length,
                                   dropout=dropout).to(device)

    encoder, decoder = load_checkpoint(
        path=os.path.join(current_path, f"model_checkpoint/{name}.pth"),
        encoder=encoder, decoder=decoder)

    encoder.eval()
    decoder.eval()

    # input_sentence = 'how are you'
    #
    # words, attention_weights = inference(encoder, decoder, input_sentence, vocab)
    #
    # visualize_attention_weights(input_sentence, words, attention_weights)

    dataset_input(encoder, decoder, vocab)
