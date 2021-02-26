# data stuff
import numpy as np
from utils.data.datasets import LineDataset
from utils.data.encoders import CharEncoder

# Neural netty stuffs
import torch
from character.networks import CharRNN, train_rnn_classifier
from torch.utils.data import DataLoader

# io file stuff
import pathlib

# string constants and stuff
import string
ALL_LETTERS = string.ascii_letters + ".,;-'"
ALL_LETTERS_ARRAY = np.array(list(ALL_LETTERS))


def input_to_vector():
    enc = CharEncoder()
    print("Input a name:")
    name = input()
    return enc.transform(name)


if __name__ == '__main__':

    # model attributes:
    vocab_size = len(ALL_LETTERS)
    hidden_size = 128
    n_classes = 18
    model = CharRNN(vocab_size, hidden_size, n_classes)

    model_path = pathlib.Path('saved_models/name_origin.pt')
    model_dir = model_path.parent

    if pathlib.Path.exists(model_path):
        model = model.load_state_dict(torch.load(model_path))
        X = input_to_vector()
        ps, idxs = model.predict(X)

        print(f"Top {len(p)} probable classes:")
        for p, idx in zip(ps, idxs):
            print(f"\tclass {idx} probability: {p}")

    else:
        print("No saved model found. Training...")
        dataset = LineDataset('../data/names/*.txt')
        trainloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
        train_rnn_classifier(model, 20, trainloader, lr=1e-4)
        print("Training complete. Saving model.")

        if pathlib.Path.exists(model_dir):
            torch.save(model.state_dict(), model_path)
        else:
            pathlib.Path.mkdir(model_dir)
            torch.save(model.state_dict(), model_path)

        print("Saved. Please re-run classification.\n")