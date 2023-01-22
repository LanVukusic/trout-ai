# create model
import torch


class FeedForward(torch.nn.Module):
    def __init__(self):
        super(FeedForward, self).__init__()
        self.fc1 = torch.nn.Linear(12 * 8 * 8 + 2, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 256)  # 2 for turn
        self.fc4 = torch.nn.Linear(256, 128)  # 2 for turn
        self.fc5 = torch.nn.Linear(128, 64)
        self.fc6 = torch.nn.Linear(64, 1)

        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.25)

    def forward(self, x):
        # reshape x to be 12*8*8
        # x = x.view(-1, 12*8*8)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc6(x)

        return x

    def predict(self, x):
        # reshape x to be 12*8*8
        # x = x.view(-1, 12*8*8)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        x = self.activation(x)
        x = self.fc5(x)
        x = self.activation(x)
        x = self.fc6(x)

        return x

    def embed(self, x):
        # reshape x to be 12*8*8
        # x = x.view(-1, 12*8*8)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.fc4(x)
        x = self.activation(x)
        x = self.fc5(x)

        return x
