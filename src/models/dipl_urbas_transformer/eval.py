import torch
import numpy as np

# # display CUDA devices available
# torch.cuda.get_device_name(0)

# set device to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print("RUNNING ON:", device)

# %%
# load data

evals = np.load("evaluations.npy")
turns = np.load("turns.npy")
one_hot_boards = np.load("one_hot_boards.npy")

# reshape data to be 2D
evals = evals.reshape(-1, 1)
# normalize data to be / 100
EVAL_DIVISOR = 1000
evals = evals / EVAL_DIVISOR

# shuffle data
np.random.seed(0)
np.random.shuffle(evals)
np.random.seed(0)
np.random.shuffle(turns)
np.random.seed(0)
np.random.shuffle(one_hot_boards)

# convert to tensors
evals = torch.from_numpy(evals).float()
turns = torch.from_numpy(turns).float()
boards = torch.from_numpy(one_hot_boards).float()

# # flatten boards and add turns
# train_boards = train_boards.view(train_boards.shape[0], -1)
# train_boards = torch.cat((train_boards, train_turns), 1)

# test_boards = test_boards.view(test_boards.shape[0], -1)
# test_boards = torch.cat((test_boards, test_turns), 1)

boards = boards.view(-1, 12, 64).transpose(1, 2)

import random

from model import Transformer

model = Transformer()
model.load_state_dict(torch.load("model.pth"))
model.eval()

for i in range(boards.shape[0]):
    board = boards[i].unsqueeze(0)
    true = evals[i]
    turn = turns[i].unsqueeze(0)

    board = torch.cat((turn.unsqueeze(1).repeat(1, 64, 1), board), dim=2)

    preds = model(board)

    print(
        f"Pred: {preds.item() * EVAL_DIVISOR / 100}      Real: {true.item() * EVAL_DIVISOR  / 100}"
    )
    input()