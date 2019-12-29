import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision

from gmcdml.app.utils import (
    imshow,
    select_n_random,
    imadd,
    get_run_path )
from gmcdml.app.cifar10 import ImageData


MODEL_PATH = './cifar_net.pth'
# default `log_dir` is "runs" - we'll be more specific here
BOARD_ROOT = 'runs/vehiclenet'

class CnvNet(nn.Module):

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(self):
        super(CnvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def images_to_probs(self, images):
        """
        Generates predictions and corresponding probabilities from a trained
        network and a list of images
        """
        output = self(images)
        # convert output probabilities to predicted class
        _, preds_tensor = torch.max(output, 1)
        preds = np.squeeze(preds_tensor.numpy())
        return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]

    def plot_classes_preds(self, images, labels):
        """
        Generates matplotlib Figure using a trained network, along with images
        and labels from a batch, that shows the network's top prediction along
        with its probability, alongside the actual label, coloring this
        information based on whether the prediction was correct or not.
        Uses the "images_to_probs" function.
        """
        preds, probs = self.images_to_probs(images)
        # plot the images in the batch, along with predicted and true labels
        fig = plt.figure(figsize=(12, 4))
        for idx in np.arange(4):
            ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
            imadd(images[idx])
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                self.classes[preds[idx]],
                probs[idx] * 100.0,
                self.classes[labels[idx]]),
                        color=("green" if preds[idx]==labels[idx].item() else "red"))
        plt.show()
        return fig

    def logStepEmbedding(self, trainset, writer, step):
        n = 100
        # select random images and their target indices
        images, labels = select_n_random(trainset.data, trainset.targets, n)

        # get the class labels for each image
        class_labels = [self.classes[lab] for lab in labels]

        # label images
        imgs = []
        for image in images:
            " Np to Tensor "
            timg = transforms.ToTensor()(image)
            imgs.append(timg)
        label_images = torch.stack(imgs)

        # feature images
        imgs = []
        idx = 0
        for image in images:
            " Np to Feature "
            fimg = torch.tensor(image)
            imgs.append(fimg)
        feature_images = torch.stack(imgs)
        features = feature_images.view(-1, 3 * 32 * 32)

        # log embeddings
        writer.add_embedding(features,
                              metadata=class_labels,
                              label_img=label_images,
                              global_step=step)
        writer.close()

    # helper function
    def add_pr_curve_tensorboard(self, class_index, test_probs, test_preds, writer, step):
        '''
        Takes in a "class_index" from 0 to 9 and plots the corresponding
        precision-recall curve
        '''
        tensorboard_preds = test_preds == class_index
        tensorboard_probs = test_probs[:, class_index]

        writer.add_pr_curve(self.classes[class_index],
                            tensorboard_preds,
                            tensorboard_probs,
                            global_step=step)
        writer.close()

    def add_precision_recall(self, testloader, writer, step):
        # 1. gets the probability predictions in a test_size x num_classes Tensor
        # 2. gets the preds in a test_size Tensor
        # takes ~10 seconds to run
        class_probs = []
        class_preds = []
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                output = self(images)
                class_probs_batch = [F.softmax(el, dim=0) for el in output]
                _, class_preds_batch = torch.max(output, 1)

                class_probs.append(class_probs_batch)
                class_preds.append(class_preds_batch)

        test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
        test_preds = torch.cat(class_preds)

        # plot all the pr curves
        for i in range(len(self.classes)):
            self.add_pr_curve_tensorboard(i, test_probs, test_preds, writer, step)

    def trainNetwork(self, iterations, trainloader, trainset, testloader, optimizer, criterion, writer):
        """ loop over the dataset multiple times """
        for epoch in range(iterations):  #

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    step = epoch * len(trainloader) + i

                    # ...log the running loss
                    writer.add_scalar('training loss',
                                      running_loss / 2000,
                                      step)

                    # ...log a Matplotlib Figure showing the model's predictions on a
                    # random mini-batch
                    writer.add_figure('predictions vs. actuals',
                                      self.plot_classes_preds(inputs, labels),
                                      global_step=step)

                    writer.close()

                    self.logStepEmbedding(trainset, writer, step)

                    self.add_precision_recall(testloader, writer, step)

                    running_loss = 0.0

        print('Finished Training')

    def testNetworkSample(self, testloader):
        dataiter = iter(testloader)
        images, labels = dataiter.next()

        # print images
        imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join('%5s' % self.classes[labels[j]] for j in range(4)))

        outputs = self(images)
        _, predicted = torch.max(outputs, 1)

        print('Predicted: ', ' '.join('%5s' % self.classes[predicted[j]] for j in range(4)))

    def testNetworkSet(self, testloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    def classAccuracy(self, testloader):
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                self.classes[i], 100 * class_correct[i] / class_total[i]))

class VehicleNet(object):

    def getWriter(self, clearruns):
        run_path = get_run_path(BOARD_ROOT, clearruns)
        return SummaryWriter(run_path)

    def setOptimizer(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001, momentum=0.9)

    def setImageData(self):
        self.imgData = ImageData()
        self.imgData.downloadImages()

    def setNetwork(self):
        self.network = CnvNet()

    def __init__(self, clearruns=False):
        self.writer = self.getWriter(clearruns)
        self.setNetwork()
        self.setImageData()
        self.setOptimizer()

    def loadNetwork(self):
        cnvnet = CnvNet()
        cnvnet.load_state_dict(torch.load(MODEL_PATH))
        self.network = cnvnet

    def saveNetwork(self):
        torch.save(self.network.state_dict(), MODEL_PATH)

    def logSampleImages(self):

        # get some random training images
        dataiter = iter(self.imgData.trainloader)
        images, labels = dataiter.next()

        # Why are we getting Tensors here?
        # print ( images.size() )
        # images.reshape(4, 32, 32, 3)
        # create grid of images
        img_grid = torchvision.utils.make_grid(images)

        imshow(img_grid)

        # write to tensorboard
        self.writer.add_image('four_vehicle_images', img_grid)
        self.writer.add_graph(self.network, images)
        self.writer.close()

    def trainAndReport(self, iterations=1):
        self.logSampleImages()
        self.network.trainNetwork(iterations,
                                  self.imgData.trainloader, self.imgData.trainset,
                                  self.imgData.testloader,
                                  self.optimizer, self.criterion,
                                  self.writer)
        self.saveNetwork()
        self.network.testNetworkSet(self.imgData.testloader)
        self.network.classAccuracy(self.imgData.testloader)

