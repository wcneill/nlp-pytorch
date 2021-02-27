# data stuff
from utils.data.datasets import LineDataset
from utils.data.encoders import CharEncoder

# Neural netty stuffs
import torch
from character.networks import CharRNN, train_rnn_classifier
from torch.utils.data import DataLoader

# io file stuff
import pathlib

# constants
import constants as cn


def input_to_tensor():
    """
    Take command line string input and encode to torch.Tensor/
    :return:
    """
    enc = CharEncoder()
    print("Input a name:", end=' ')
    name = input()
    array = enc.transform(name)
    tensor = torch.tensor(array).to(cn.DEVICE)
    tensor = tensor.unsqueeze(0).float()
    return tensor


def infer(model, X, top=3):
    """
    Perform inference using given model.
    :param model: The model to infer with.
    :param X: The input tensor.
    :param top: Determines how many results are returned.
    :return: (probs, idxs) tuple of probabilities and indexes of top k most likely classes.
    """

    ps, idxs = model.predict(X, top_k=top)
    ps, idxs = ps.squeeze(), idxs.squeeze()

    print(f"Top {top} probable classes:")
    for p, idx in zip(ps, idxs):
        print(f"\t{cn.ORIGIN_CLASSES[idx]} probability: {p:.3f}")

    return ps, idxs


def train(model, dataloader):

    train_rnn_classifier(model, 20, dataloader, lr=1e-4)
    print("Training complete. Saving model.")

    if pathlib.Path.exists(model_dir):
        torch.save(model.state_dict(), model_path)
    else:
        pathlib.Path.mkdir(model_dir)
        torch.save(model.state_dict(), model_path)


if __name__ == '__main__':

    # model attributes:
    vocab_size = len(cn.ALL_LETTERS)
    hidden_size = 128
    n_classes = 18
    model = CharRNN(vocab_size, hidden_size, n_classes)
    model = model.to(cn.DEVICE)

    model_path = pathlib.Path('saved_models/name_origin.pt')
    model_dir = model_path.parent

    if pathlib.Path.exists(model_path):

        model.load_state_dict(torch.load(model_path))
        print(f"Saved model found and loaded: \n{model}\n")
        X = input_to_tensor()
        infer(model, X)

    else:
        print("No saved model found. Training...")
        dataset = LineDataset('../data/names/*.txt')
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
        train(model, loader)
        print("Model trained and saved. Please re-run classification.\n")