# File handling
from io import open
import glob
import string
import pathlib

# Data and such and things
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Neural netty stuffs
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

# string constants
ALL_LETTERS = string.ascii_letters + ".,;'"
ALL_LETTERS_ARRAY = np.array(list(ALL_LETTERS))


def train_rnn(model: 'RNN', X_train, y_train, epochs, lr=1e-3, criterion=nn.NLLLoss(), opt=Adam):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training model on {device}')
    model = model.to(device)

    opt = opt(model.parameters(), lr=lr)
    hs = model.init_hidden().to(device)

    for e in range(epochs):
        opt.zero_grad()


def train_one_epoch(model, opt, criterion, X_train, y_train):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for line, target in zip(X_train, y_train):
        probs, hs = model.get_probs(line, hs)
        loss = criterion(probs, target)
        loss.backward()
        opt.step()


class LineDataset(Dataset):
    """
    A dataset created from a file structure in the format `<your dir>/<target category>.txt`.

    To create this dataset, pass a glob pattern to the constructor in the form
    `glob_pattern='path/to/data/*.txt'`
    """
    def __init__(self, glob_pattern):
        self.data = self.df_from_glob(glob_pattern)
        self.encoder = WordEncoder()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        raw_data = self.data.iloc[idx].to_numpy()

    def df_from_glob(self, glob_pattern):
        """
        Create dataframe of samples where `columns=['Line', 'Category']`

        :param glob_pattern: Pattern for recursively searching directories for text files
            in the format `<target>.txt` where each .txt file contains one instance of that
            target category per line. Example: `glob_pattern="data/names/*.txt"
        :return: dataframe of samples described above.
        """
        files = self.find_files(glob_pattern)

        dataframe = pd.DataFrame(columns=['Line', 'Category'])
        for cat_file in files:
            # get category from file name
            category = pathlib.Path(cat_file).stem

            # get each instance (one per line in file) into a list.
            lines = self.lines_to_list(cat_file)

            # create dataframe from this file, and concatenate it to the dataset
            data = pd.DataFrame.from_dict({line: category for line in lines}, orient='index')
            pd.concat(dataframe, data)

        return dataframe


    @staticmethod
    def find_files(path_pattern):
        return glob.glob(path_pattern)

    @staticmethod
    def lines_to_list(filepath, encoding='utf-8'):
        "Read and return each line with trailing newlines stripped."
        with open(filepath, encoding=encoding) as f:
            lines = f.readlines()
            return [l.rstrip() for l in lines]


class WordEncoder:
    def __init__(self):
        self.encoder = OneHotEncoder(sparse=False).fit(ALL_LETTERS_ARRAY.reshape(-1, 1))

    def transform(self, word):
        word = np.array(list(word)).reshape(-1, 1)
        return self.encoder.transform(word).reshape(len(word), 1, -1)

    def inverse_transform(self, X):
        word_arr = self.encoder.inverse_transform(X).reshape(-1,)
        return ''.join(word_arr)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size

        self.in2out = nn.Linear(input_size + hidden_size, output_size)
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hs):
        combined = torch.cat((input, hs), dim=1)
        hidden = self.in2hidden(combined)
        out = self.in2out(combined)
        out = self.softmax(out)

        return out, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

    def get_probs(self, X, hs):
        for x in X:
            output, hs = self.forward(x, hs)
        return output


if __name__ == '__main__':

    dt = np.dtype([('string', 'U29')])
    print(dt['string'])

    x = np.array([], dtype=dt)
    np.append(x, 'asdfg')
