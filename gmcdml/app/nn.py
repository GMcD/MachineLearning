import torch
import torch.nn as nn
import torch.optim as optim
from app.net import Net


def sampleNet():
    net = Net()
    print(net)
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())  # conv1's .weight


def sampleOutput():
    net = Net()
    input = torch.randn(1, 1, 32, 32)
    out = net(input)
    print(out)
    return (net, out)


def resetNN():
    net, out = sampleOutput()
    net.zero_grad()
    out.backward(torch.randn(1, 10))


def mseLoss():
    net = Net()
    input = torch.randn(1, 1, 32, 32)
    output = net(input)
    target = torch.randn(10)  # a dummy target, for example
    target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss()

    loss = criterion(output, target)
    print(loss)
    print(loss.grad_fn)  # MSELoss
    print(loss.grad_fn.next_functions[0][0])  # Linear
    print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
    return (net, loss)

def backPropagate():
    net, loss = mseLoss()
    net.zero_grad()  # zeroes the gradient buffers of all parameters

    print('conv1.bias.grad before backward')
    print(net.conv1.bias.grad)

    loss.backward()

    print('conv1.bias.grad after backward')
    print(net.conv1.bias.grad)

def updateWeights():
    net = Net()
    input = torch.randn(1, 1, 32, 32)

    # create your optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # in your training loop:
    optimizer.zero_grad()  # zero the gradient buffers
    output = net(input)
    target = torch.randn(10)  # a dummy target, for example
    target = target.view(1, -1)  # make it the same shape as output
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()  # Does the update
