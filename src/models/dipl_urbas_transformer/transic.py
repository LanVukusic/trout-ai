# %%
import torch
import numpy as np

# # display CUDA devices available
# torch.cuda.get_device_name(0)

# set device to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

# split data into train and test
SPLIT_INDEX = int(len(evals) * 0.8)
train_evals = evals[:SPLIT_INDEX]
train_turns = turns[:SPLIT_INDEX]
train_boards = one_hot_boards[:SPLIT_INDEX]

test_evals = evals[SPLIT_INDEX:]
test_turns = turns[SPLIT_INDEX:]
test_boards = one_hot_boards[SPLIT_INDEX:]

# convert to tensors
train_evals = torch.from_numpy(train_evals).float()
train_turns = torch.from_numpy(train_turns).float()
train_boards = torch.from_numpy(train_boards).float()

test_evals = torch.from_numpy(test_evals).float()
test_turns = torch.from_numpy(test_turns).float()
test_boards = torch.from_numpy(test_boards).float()

# # flatten boards and add turns
# train_boards = train_boards.view(train_boards.shape[0], -1)
# train_boards = torch.cat((train_boards, train_turns), 1)

# test_boards = test_boards.view(test_boards.shape[0], -1)
# test_boards = torch.cat((test_boards, test_turns), 1)

train_boards = train_boards.view(-1, 12, 64).transpose(1, 2)
test_boards = test_boards.view(-1, 12, 64).transpose(1, 2)

print(train_boards.shape)

# # print 1 example
# print(train_evals[6])
# print(train_turns[0])
# print(train_boards[0])

# %%
# train model
from model import Transformer

loss = torch.nn.L1Loss()
model = Transformer().to(device)
print("Model loaded")
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

# %%
# model graphing code
test_loss_data = []
train_loss_data = []

# hyperparameters
EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-5

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# train
for epoch in range(EPOCHS):
    for i in range(0, len(train_evals), BATCH_SIZE):
        # get batch
        batch_evals = train_evals[i:i + BATCH_SIZE].to(device)
        batch_boards = train_boards[i:i + BATCH_SIZE].to(device)
        batch_turns = train_turns[i:i + BATCH_SIZE].to(device)

        batch_boards = torch.cat(
            (batch_turns.unsqueeze(1).repeat(1, 64, 1), batch_boards), dim=2)

        # forward pass
        y_pred = model(batch_boards)

        # calculate loss
        loss_val = loss(y_pred, batch_evals)

        # backward pass
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    print(f"{epoch}: {loss_val}")

    # print loss every 100 epochs
    if epoch % 10 == 0:
        with torch.inference_mode():
            # use test data to calculate loss
            test_losses = []
            for i in range(0, len(test_boards), BATCH_SIZE):
                boards = test_boards[i:i + BATCH_SIZE]
                target = test_evals[i:i + BATCH_SIZE]
                turns = test_turns[i:i + BATCH_SIZE]

                boards = torch.cat(
                    (turns.unsqueeze(1).repeat(1, 64, 1), boards), dim=2)

                y_pred = model(boards.to(device))
                test_losses.append(loss(y_pred, target.to(device)).item())

            # use train data to calculate loss
            # y_pred = model(train_boards.to(device))
            # train_loss = loss(y_pred, train_evals.to(device)).item() * EVAL_DIVISOR
            test_loss = np.mean(test_losses)
            print("Epoch: ", epoch, "test: ", np.mean(test_loss))

            # save loss data
            test_loss_data.append(test_loss)
            #train_loss_data.append(train_loss)

import matplotlib.pyplot as plt

torch.save(model.state_dict(), "model.pth")
# plot loss
from datetime import datetime

plt.plot(test_loss_data, label="test")
plt.plot(train_loss_data, label="train")
plt.legend()
plt.show()

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
OUT_NAME = "IMG/trans-{}-loss.png".format(current_time)
plt.savefig(OUT_NAME)
