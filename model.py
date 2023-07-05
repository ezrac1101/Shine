import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(784, 256))
        self.fc.append(nn.Linear(256, 256))
        self.fc.append(nn.Linear(256, 10))

    def forward(self, x):
        #  print("input: {}".format(x))
        x = torch.flatten(x, start_dim=1)
        for idx, layer in enumerate(self.fc):
            #  print("layer {}: {}".format(idx, x))
            x = layer(x)
            if idx != len(self.fc) - 1:
                x = F.relu(x)
        output = F.log_softmax(x, dim=1)
        return output

device = torch.device("cpu")
def main():
    transform=transforms.Compose([transforms.ToTensor()])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform = transform)
    dataset2 = datasets.MNIST('../data', train=False, transform = transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size = 1000)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size = 1000)

    model = Net().to(device)
    for l in model.fc:
        print("in: {}, out: {}".format(l.in_features, l.out_features))
    print(len(model.fc))
    learning_rate = 1
    print (learning_rate)
    optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)

    num_epoch = 14
    print (num_epoch)
    for epoch in range(1, num_epoch):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        print (data[0,0].shape)
        plt.imshow(data[0,0])
        plt.savefig('foo.png')
        input()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step() #new set of gradients propogated back into each of the networks
        if batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    import time
    start = time.time()
    main()
    print (time.time()- start)