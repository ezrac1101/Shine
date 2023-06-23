import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(2, 1))
        self.fc.append(nn.Linear(1, 3))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        for idx, layer in enumerate(self.fc):
            x = layer(x)
            #  if idx != len(self.fc) - 1:
            #      x = F.relu(x)
        return x


if __name__ == '__main__':
    network = Net()
    network.fc[0].weight = nn.Parameter(torch.FloatTensor([[0, 1]]))
    network.fc[0].bias = nn.Parameter(torch.FloatTensor([0.3]))
    network.fc[1].weight = nn.Parameter(torch.FloatTensor([[1], [2], [3]]))
    network.fc[1].bias = nn.Parameter(torch.FloatTensor([0.001, -1, 0]))

    x = torch.tensor([[0.1,0.2]])
    #  x = torch.randn(batch_size, input_dim, requires_grad=True)
    print(x)
    print(network(x))
