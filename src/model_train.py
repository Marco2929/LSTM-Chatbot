from __future__ import unicode_literals, print_function, division

import os
import pickle
import time
import math

from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from src.model_inference import load_checkpoint
from src.model_prepare_dataset import prepare_dataset, get_dataloader, Vocab

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_p=0.1):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout_p if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.lstm(embedded)
        return output, hidden


class BahdanauAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttentionLayer, self).__init__()
        self.query_layer = nn.Linear(hidden_size, hidden_size)  # Wa
        self.key_layer = nn.Linear(hidden_size, hidden_size)  # Ua
        self.score_layer = nn.Linear(hidden_size, 1)  # Va

    def forward(self, query, keys):
        # Ensure query has shape (batch_size, 1, hidden_size)
        query = query.unsqueeze(1)  # Add time dimension

        # Apply linear transformations
        transformed_query = self.query_layer(query)  # Shape: (batch_size, 1, hidden_size)
        transformed_keys = self.key_layer(keys)  # Shape: (batch_size, seq_len, hidden_size)

        # Compute scores
        scores = self.score_layer(torch.tanh(transformed_query + transformed_keys))  # Shape: (batch_size, seq_len, 1)
        scores = scores.squeeze(2).unsqueeze(1)  # Shape: (batch_size, 1, seq_len)

        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)  # Shape: (batch_size, 1, seq_len)

        # Compute context vector
        context_vector = torch.bmm(attention_weights, keys)  # Shape: (batch_size, 1, hidden_size)

        return context_vector, attention_weights


class DecoderLSTMAttention(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, num_layers=1, dropout=0.1):
        super(DecoderLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=0)
        self.attention = BahdanauAttentionLayer(hidden_size)
        self.lstm = nn.LSTM(2 * hidden_size, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(self.max_length):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions

    def forward_step(self, input, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input))

        query = hidden[0][-1]  # Take the last layer's hidden state from (h_n, c_n)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_lstm = torch.cat((embedded, context), dim=2)

        output, hidden = self.lstm(input_lstm, hidden)
        output = self.out(output)

        return output, hidden, attn_weights


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def train_epoch(train_dataloader: DataLoader, encoder: EncoderLSTM, decoder: DecoderLSTMAttention, clip: float,
                encoder_optimizer: optim,
                decoder_optimizer: optim, criterion: nn.NLLLoss):
    total_loss = 0
    for data in train_dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)

        decoder_outputs, _, _ = decoder(encoder_outputs=encoder_outputs, encoder_hidden=encoder_hidden,
                                        target_tensor=target_tensor)

        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        nn.utils.clip_grad_norm_(decoder.parameters(), clip)

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_dataloader)


def train(train_dataloader: DataLoader, test_dataloader: DataLoader, vocab: Vocab, encoder: EncoderLSTM,
          decoder: DecoderLSTMAttention, encoder_optimizer: optim.Adam, decoder_optimizer: optim.Adam, clip: float,
          epochs: int,
          print_every=100):
    start = time.time()
    print_loss_total = 0

    criterion = nn.NLLLoss()

    # Define a timestamp for the logging directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create a directory for the run logs
    log_dir = f'runs/{timestamp}_ep_{epochs}'
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize the SummaryWriter with the logging directory
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(1, epochs + 1):
        loss = train_epoch(train_dataloader=train_dataloader, encoder=encoder, decoder=decoder,
                           encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer, clip=clip,
                           criterion=criterion)
        print_loss_total += loss

        # Log the training loss to TensorBoard
        writer.add_scalar('train/loss', loss, epoch)

        # Evaluate on the test set
        test_loss = test_epoch(test_dataloader=test_dataloader, encoder=encoder, decoder=decoder, criterion=criterion)

        # Log the test loss to TensorBoard
        writer.add_scalar('test/loss', test_loss, epoch)

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) Train Loss: %.4f | Test Loss: %.4f' % (time_since(start, epoch / epochs),
                                                                       epoch, epoch / epochs * 100,
                                                                       print_loss_avg, test_loss))
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(epoch, encoder, decoder, loss, vocab, path=checkpoint_path)

        # Close the TensorBoard writer
        writer.close()


def test_epoch(test_dataloader: DataLoader, encoder: EncoderLSTM, decoder: DecoderLSTMAttention, criterion: nn.NLLLoss):
    total_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            input_tensor, target_tensor = data

            encoder_outputs, encoder_hidden = encoder(input_tensor)

            decoder_outputs, _, _ = decoder(encoder_outputs=encoder_outputs, encoder_hidden=encoder_hidden,
                                            target_tensor=target_tensor)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)),
                target_tensor.view(-1)
            )
            total_loss += loss.item()

    return total_loss / len(test_dataloader)


def save_checkpoint(epoch: int, encoder: EncoderLSTM, decoder: DecoderLSTMAttention, loss: float, vocab: Vocab,
                    path: str):
    checkpoint = {
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)

    with open(Path(path).with_suffix('.pkl'), 'wb') as file:
        pickle.dump(vocab, file)


def load_vocab(vocab_path: str) -> Vocab:
    with open(vocab_path, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


if __name__ == '__main__':
    # Configure training/optimization
    learning_rate = 0.0001
    epochs = 100
    clip = 50.0
    hidden_size = 500
    encoder_layer_number = 2
    decoder_layer_number = 2
    batch_size = 512
    max_length = 15
    dropout = 0.1

    vocab, train_pairs, test_pairs = prepare_dataset(max_length=max_length)

    encoder = EncoderLSTM(input_size=vocab.num_of_words, num_layers=encoder_layer_number, hidden_size=hidden_size).to(
        device)
    decoder = DecoderLSTMAttention(hidden_size=hidden_size, output_size=vocab.num_of_words,
                                   num_layers=decoder_layer_number,
                                   max_length=max_length,
                                   dropout=dropout).to(device)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    train_dataloader, test_dataloader = get_dataloader(vocab=vocab, batch_size=batch_size, max_length=max_length,
                                                       train_pairs=train_pairs, test_pairs=test_pairs)

    train(train_dataloader=train_dataloader, test_dataloader=test_dataloader, vocab=vocab,
          encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer, clip=clip, encoder=encoder,
          decoder=decoder, epochs=epochs, print_every=10)
