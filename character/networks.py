
# Data and such and things
import copy
import numpy as np
from utils.data.datasets import LineDataset

# Neural netty stuffs
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

# Progress bar
from tqdm import tqdm

# string constants and stuff
import constants as cn


def train_rnn_classifier(model, epochs, trainloader,
                         testloader=None, criterion=nn.NLLLoss(), optimizer=optim.Adam, lr=1e-4, es_patience=20):
    """
    Trains an RNN for classification. Assumes that the output of the model are class probabilities.

    This trainer makes the assumptions:

        1. that each sequence is separate for the last, so the hidden
        state is not preserved between sequences. An optional flag may be added in a later version
        that will cause the hidden state to be preserved for an entire epoch.

        2. Your model outputs a tuple (probs, lstm_out, lstm_hidden), which is the conventions of models
        in this package. `probs` is the result of  LogSoftmax applied to the final rnn hidden state, and ``lstm_out``
        and ``lstm_hidden`` are the raw outputs of the RNN module.

    :param model: Your RNN.
    :param epochs: Number of times the RNN will see each piece of data.
    :param trainloader: A PyTorch DataLoader object
    :param testloader: Optional. A validation dataset wrapped in a second PyTorch DataLoader
    :param criterion: Loss function. Default ``nn.LLLoss``
    :param optimizer: Optimizer. Default Adam.
    :param lr: Learning rate. Default 1e-4.
    :param es_patience: Early stopping patience. Works in conjunction with a validation
        dataset to monitor generalized performance and stop training before overfitting occurs.
        Must supply the ``testloader`` argument with a DataLoader object to use this functionality.
    :return: Loss history in a tuple of lists: ``(training, validation)``. If no validation data
        is supplied to the trainer, validation loss is an empty list.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training model on {device}')
    model = model.to(device)

    train_loss = []
    valid_loss = []
    best_loss = np.inf
    best_model_wts = None

    curr_patience = es_patience

    opt = optimizer(model.parameters(), lr=lr)

    for e in tqdm(range(epochs)):
        tl, vl = train_one_epoch(model, opt, criterion, trainloader, testloader)

        if testloader is not None:
            valid_loss.append(vl / len(testloader))
            if valid_loss[-1] < best_loss:
                best_loss = valid_loss[-1]
                best_model_wts = copy.deepcopy(model.state_dict())
                curr_patience = es_patience
            else:
                curr_patience -= 1

        if curr_patience == 0:
            print(f'No improvement in {es_patience} epochs. Interrupting training.')
            print(f'Best loss: {best_loss}')
            print(f'Loading best model weights.')
            model.load_state_dict(best_model_wts)
            print('Training complete.')
            break

    return train_loss, valid_loss


def train_one_epoch(model, opt, criterion, trainload, testload):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    running_tl = 0
    running_vl = 0

    hs = None

    for x, y in trainload:

        # inference
        x = x.to(device).float()
        y = y.to(device)
        probs, out, hs = model(x)

        # back-prop
        loss = criterion(probs, y)
        loss.backward()
        opt.step()
        running_tl += loss.item()

        # clear gradients
        opt.zero_grad()


    if testload is not None:
        model.eval()
        with torch.no_grad():
            for x in testload:
                x = x.to(device).float()
                loss = criterion(model(x), x)
                running_vl += loss.item()
        model.train()

    return running_tl, running_vl








class CharRNN(nn.Module):
    """
    A basic many to one RNN model for set up classification or regression tasks.

    This model returns three tensors:

       1. Softmax(hs_N), where hs_N is the final hidden state of the RNN ``hs[-1]`` (batch_size, output_size)
       2. RNN final layer output (batch_size, seq_length, hidden_size)
       3. RNN final hidden state (num_layers, batch_size, hidden_size)

    This allows for flexibility in training the network for different kinds of
    tasks. The Softmax output is ideal for classification tasks, where-as the
    LSTM output may be used for seq-to-seq, classification or regression tasks.

    Parameters:
        :param input_size: The number of variables in the sequence.
        :param hidden_size: The number of variables in the latent space
        :param output_size: Output size of network.

    """
    def __init__(self, input_size, hidden_size, output_size, n_layers=2, do=0.2):
        super().__init__()

        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size, num_layers=n_layers, batch_first=True, dropout=do)
        self.dense = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hs=None):
        # print(" -------- In forward ----------")
        # print(f"Input shape: {input.shape}")
        out, hs = self.rnn(input, hs)
        # print(f"rnn output shape: {out.shape}")
        # print(f"rnn hidden shape: {hs[-1].shape}")
        probs = self.dense(hs[-1])
        # print(f"dense layer out {out.shape}")
        probs = self.softmax(probs)
        # print(f"Softmax out: {(torch.exp(probs).sum(dim=1))}")

        return probs, out, hs

    def predict(self, input, top_k=3):
        log_probs, _, _ = self(input)
        probs = torch.exp(log_probs)
        return probs.topk(top_k)


if __name__ == '__main__':

    dataset = LineDataset('../data/names/*.txt')
    trainloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    n_classes = len(dataset.class_codes)
    vocab_size = len(cn.ALL_LETTERS)
    print(f"Number of classes: {n_classes}")
    print(f"Vocabulary size: {vocab_size}\n")

    hidden_size = 128
    model = CharRNN(vocab_size, hidden_size, n_classes)

    train_rnn_classifier(model, 1, trainloader, lr=1e-4)

