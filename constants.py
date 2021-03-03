import numpy as np
import torch
import string
import pathlib

# Not so constant:
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Constants for name origin class tasks
ALL_LETTERS = string.ascii_letters + ".,;'-/"  # / serves as EOS marker.
ALL_LETTERS_ARRAY = np.array(list(ALL_LETTERS))
ORIGIN_CLASSES = ['Czech', 'German', 'Arabic', 'Japanese',
                  'Chinese', 'Vietnamese', 'Russian', 'French',
                  'Irish', 'English',  'Spanish', 'Greek', 'Italian',
                  'Portuguese', 'Scottish', 'Dutch', 'Korean', 'Polish']
NUM_CLASSES = len(ORIGIN_CLASSES)
