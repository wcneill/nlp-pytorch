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

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
        tl, vl = _train_one_epoch(model, opt, criterion, trainloader, testloader)

        train_loss.append(tl / len(trainloader))

        if testloader is not None:
            valid_loss.append(vl / len(testloader))
            if valid_loss[-1] < best_loss:
                best_loss = valid_loss[-1]
                best_model_wts = copy.deepcopy(model.state_dict())
                curr_patience = es_patience
            else:
                curr_patience -= 1

        if curr_patience == 0:
            _load_best(model, best_loss, best_model_wts)
            break

    return train_loss, valid_loss


def _load_best(model, best_loss, best_weights):
    """
    Helper function for early stopping and loading best weights.
    """
    print(f'Early stopping patience reached. Interrupting training.')
    print(f'Best loss: {best_loss}')
    print(f'Loading best model weights.')
    model.load_state_dict(best_weights)
    print('Training complete.')


def _train_one_epoch(model, opt, criterion, trainload, testload):
    """Train a model for a single epoch"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    running_tl = 0
    running_vl = 0

    model.train()
    for x, y in trainload:

        # inference
        x = x.to(device)#.float()
        y = y.to(device)
        probs, out, hs = model(x)

        # back-prop
        loss = criterion(probs, y)
        loss.backward()
        opt.step()
        running_tl += loss.item()

        # clear gradients
        opt.zero_grad()

    # Gather validation error metrics, if available
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
        :param vocab_size: Length of vocabulary.
        :param emb_size: Size of latent space
        :param hidden_size: The number of variables in the latent space
        :param output_size: Output size of network.
        :param n_layers: Default 2. Number of RNN layers to stack.
        :param do: Dropout percent in [0, 1).

    """
    def __init__(self, vocab_size, emb_size, hidden_size, output_size, n_layers=2, do=0.2):
        super().__init__()

        self.hidden_size = hidden_size

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.RNN(emb_size, hidden_size, num_layers=n_layers, batch_first=True, dropout=do)
        self.dense = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hs=None):
        """forward prop. Returns softmax of final hidden state, the output of the final layer
        and the final hidden state."""
        inputs = self.emb(inputs)
        out, hs = self.rnn(inputs, hs)
        probs = self.dense(hs[-1])
        probs = self.softmax(probs)

        return probs, out, hs

    def predict(self, inputs, top_k=3):
        """Performs inference by returning the top_k most likely classes according to the network."""
        log_probs, _, _ = self(inputs)
        probs = torch.exp(log_probs)
        return probs.topk(top_k)


class CharGenNet(CharRNN):
    """
    An RNN that can be trained to generate text, one character at a time based on an initial classification and
    starting character.

    Parameters:
        :param num_classes: The number of classes that generated text can belong to.
        :param vocab_size: Length of vocabulary.
        :param emb_size: Size of latent space
        :param hidden_size: The number of variables in the latent space
        :param output_size: Output size of network.
        :param n_layers: Default 2. Number of RNN layers to stack.
        :param do: Dropout percent in [0, 1).

    """

    def __init__(self, num_classes, vocab_size, emb_size, hidden_size, output_size, **kwargs):
        super().__init__(vocab_size, emb_size, hidden_size, output_size, **kwargs)

        self.hs_emb = nn.Embedding(num_classes, hidden_size)

    # overrides base class forward.
    def forward(self, class_idx, first_char, hs=None):
        """
        Forward prop. Takes a class index and a starting character, outputs a probability distribution
        of the character at the next position.

        :param class_idx: The integer index describing the specific class from which to generate text.
            Should be an integer from [0, C - 1] where ``C`` is the number of unique classes.
        :param first_char: The starting character.

        """

        hs0_emb = self.hs_emb(class_idx)
        char_emb = self.emb(first_char)
        out, hs = self.rnn(char_emb, hs0_emb)
        out = out.reshape(-1, self.hidden_size)

        return self.softmax(out)

    def sample(self, class_idx, char, top_k=3):
        """
        Input a class and starting character, and sample the model's distribution for the next character.

        :param class_idx: Index of the starting class.
        :param char: Starting character.
        :param top_k: The sampler will get the ``top_k`` most likely next characters and their probabilities.
        It will use those probabilities to choose and return one of the ``top_k`` subset. If ``top_k=None``,
        the sampler will choose a character based on the entire distribution.
        :return: The vocabulary index of the sampled next character.
        """
        probs = self.forward(class_idx, char)

        if top_k is not None:
            probs, idxs = probs.topk(top_k)
            choice = np.random.choice(idxs, p=probs)
        else:
            choice = np.random.choice(range(len(probs)), p=probs)

        return choice


def plot_confusion(y_true, y_pred, normalize='pred'):
    """
    Create, plot and return a confusion matrix. Matrix normalized by predicted class counts, so diagonal represent
    class precision. Pass ``normalize='true'`` to get recall along diagonal.

    :param y_true: True class
    :param y_pred: Predicted class.
    :param normalize: One of ['pred', 'true']. Normalize by predicted values (generates precision on diagonal)
        or normalize by true values (generates recall on the diagonal).
    :return: Confusion matrix ndarray
    """

    m = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=cn.ORIGIN_CLASSES, normalize=normalize)
    sns.heatmap(m)
    plt.xticks(range(cn.NUM_CLASSES), labels=cn.ORIGIN_CLASSES, rotation=45)
    plt.yticks(np.arange(0.5, cn.NUM_CLASSES + 0.5, 1), labels=cn.ORIGIN_CLASSES, rotation=0)
    plt.show()

    return m


def main_classify():

    # Load the data
    dataset = LineDataset('../data/names/*.txt')
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # Train and save the network
    epochs = 20
    vocab_size = 57
    emb_size = 30
    hidden_size = 128
    output_size = 18
    rnn = CharRNN(vocab_size, emb_size, hidden_size, output_size, n_layers=2)
    loss = train_rnn_classifier(rnn, epochs, loader)
    torch.save(rnn.state_dict(), '../tasks/saved_models/name_origin_wemb.pt')

    # plot training loss
    plt.plot(loss[0])
    plt.show()

    # Generate confusion matrix:
    rnn.load_state_dict(torch.load('../tasks/saved_models/name_origin_wemb.pt'))

    true, pred = [], []
    idx_to_label = {i: v for i, v in enumerate(cn.ORIGIN_CLASSES)}

    print("Generating Confusion Matrix.")
    for X, y in tqdm(dataset):
        _, idx = rnn.predict(X.unsqueeze(0), top_k=1)
        true.append(idx_to_label[y.item()])
        pred.append(idx_to_label[idx.item()])

    m = plot_confusion(true, pred)

    print('Accuracy:', end=' ')
    print("{:.2f}".format(sum([1 if t == p else 0 for t, p in zip(true, pred)]) / len(true)))
    print(f'Avg Precision: {m.diagonal().mean():.2f}')

def main_gen():
    epochs = 20
    num_classes =
    vocab_size = 57
    emb_size = 30
    hidden_size = 128

    rnn = CharGenNet()

if __name__ == '__main__':
    main_classify()
