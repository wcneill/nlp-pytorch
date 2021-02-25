# File handling
from io import open
import glob
import string
import pathlib

# Data and such and things
import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Neural netty stuffs
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

# string constants and stuff
import unicodedata
ALL_LETTERS = string.ascii_letters + ".,;-'"
ALL_LETTERS_ARRAY = np.array(list(ALL_LETTERS))


def train_rnn_classifier(model, epochs, trainloader,
                         testloader=None, criterion=nn.NLLLoss(), optimizer=optim.Adam, lr=1e-6, es_patience=20):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training model on {device}')
    model = model.to(device)

    train_loss = []
    valid_loss = []
    best_loss = np.inf
    best_model_wts = None

    curr_patience = es_patience

    opt = optimizer(model.parameters(), lr=lr)

    for e in range(epochs):
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
        # print(x)
        x = x.to(device).float()
        opt.zero_grad()
        y_hat, hs = model(x, hs)

        # back-prop
        print("at backprop")
        print(y, y_hat)
        loss = criterion(y_hat, y)
        loss.backward()
        opt.step()
        running_tl += loss.item()

    if testload is not None:
        model.eval()
        with torch.no_grad():
            for x in testload:
                x = x.to(device).float()
                loss = criterion(model(x), x)
                running_vl += loss.item()
        model.train()

    return running_tl, running_vl


class LineDataset(Dataset):
    """
    A dataset created from a file structure in the format `<your dir>/<target class>.txt`. Where
    <target class>.txt contains a list of instances that belong to that class, one per line.

    Each instance is vectorized (one-hot encoded) at the character level, and returned in a tuple
    along with it's class index.

    To create this dataset, pass a glob pattern to the constructor in the form
    `glob_pattern='path/to/data/*.txt'`.

    Attributes:

        :ivar data: A pandas dataframe with two columns 'Line' and 'Class'. The relationship
            is that 'Line' is an instance of the 'Class' category.
        :ivar encoder: A WordEncoder which vectorizes text at the character level.
        :ivar class_codes: A dictionary containing integer encoding of each target class from
            `0, ..., N-1` where N is the number of classes.
    """
    def __init__(self, glob_pattern):
        self.data = self.df_from_glob(glob_pattern)
        self.encoder = CharEncoder()
        self.class_codes = {category: i for i, category in enumerate(self.data.Class.unique())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        x, y = self.data.iloc[idx, 0], self.data.iloc[idx, 1]
        print(x)
        X = self.encoder.transform(x)
        print(f"in dataset.__getitem__: {type(X)}")
        print(X.shape)
        y = self.class_codes[y]

        return torch.tensor(X), torch.tensor(y)

    def df_from_glob(self, glob_pattern):
        """
        Create dataframe of samples where `columns=['Line', 'Category']`

        :param glob_pattern: Pattern for recursively searching directories for text files
            in the format `<target>.txt` where each .txt file contains one instance of that
            target category per line. Example: `glob_pattern="data/names/*.txt"
        :return: dataframe of samples described above.
        """
        files = glob.glob(glob_pattern)

        dataframe = None
        for cat_file in files:

            # get category from file name
            category = pathlib.Path(cat_file).stem

            # get each instance (one per line in file) into a list.
            lines = self._lines_to_list(cat_file)
            class_dict = {line: category for line in lines}

            # create dataframe from this file, and concatenate it to the dataset
            data = pd.DataFrame({'Line': class_dict.keys(), 'Class': class_dict.values()})
            dataframe = pd.concat((dataframe, data), ignore_index=True)

        return dataframe

    def _lines_to_list(self, filepath, encoding='utf-8'):
        "Read and return each line with trailing newlines stripped."
        with open(filepath, encoding=encoding) as f:
            lines = f.readlines()
            return [self._unicode_to_ascii(l).rstrip().lstrip() for l in lines]

    @staticmethod
    def _unicode_to_ascii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in ALL_LETTERS
        )


class CharEncoder:
    def __init__(self):
        self.encoder = OneHotEncoder(sparse=False).fit(ALL_LETTERS_ARRAY.reshape(-1, 1))
        self.categories = self.encoder.categories_[0].tolist()

    def transform(self, word):
        word = np.array(list(word)).reshape(-1, 1)
        return self.encoder.transform(word).reshape(len(word), -1)

    def inverse_transform(self, X):
        word_arr = self.encoder.inverse_transform(X).reshape(-1,)
        return ''.join(word_arr)


class chaRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=2, do=0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = nn.RNN(input_size, hidden_size, num_layers=n_layers, dropout=do)
        self.dense = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hs=None):
        out, hs = self.rnn(input, hs)
        out.reshape(-1, self.hidden_size)
        out = self.dense(out)
        out = self.softmax(out)

        return out, hs

    def predict(self, input, top_k=3):
        all_probs = self(input)
        return all_probs.topk(top_k)


if __name__ == '__main__':

    dataset = LineDataset('../data/names/*.txt')
    trainloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    n_classes = len(dataset.class_codes)
    print(f"Number of classes: {n_classes}")

    print(string.ascii_letters)

    hidden_size = 128
    model = chaRNN(len(ALL_LETTERS), hidden_size, n_classes)

    train_rnn_classifier(model, 3, trainloader, lr=1e-4)

