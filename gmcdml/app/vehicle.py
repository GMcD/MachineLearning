import matplotlib.pyplot as plt
import numpy as np
import os, shutil, glob
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
    imadd )
from gmcdml.app.cifar10 import ImageData

" Location of saved models "
MODEL_PATH = 'models/vehiclenet'
" model file name "
MODEL_NAME = 'vehiclenet.pth'
" TensorBoard run data"
BOARD_ROOT = 'runs/vehiclenet'
" Training Set Size "
TRAINING_SET = 10000

class ModelState(object):
    last_model_path : str
    last_model_short : str
    this_model_path : str
    next_run_path : str
    run : int
    step : int

    def __str__(self):
        return "Next Run : {} at ({}, {})\n   From  : {}\n   To    : {}"\
            .format(self.next_run_path, self.run, self.step, self.last_model_path, self.this_model_path)

class State(object):

    def content_dir(self) -> str:
        if os.path.exists("/content"):
            return "/content"
        else:
            return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    def __init__(self):
        self.root = self.content_dir()
        self.model_dir = os.path.join(self.root, MODEL_PATH)
        self.run_dir = os.path.join(self.root, BOARD_ROOT)

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.run_dir):
            os.makedirs(self.run_dir)

    def clear(self):
        " Remove all state, and recreate empty folders "
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)
        os.makedirs(self.model_dir)
        if os.path.exists(self.run_dir):
            shutil.rmtree(self.run_dir)
        os.makedirs(self.run_dir)

    def last_run(self) -> int:
        past_runs = os.listdir(self.run_dir)
        if len(past_runs) == 0:
            return 0
        run_ids = [int(r.split('/')[-1:][0]) for r in past_runs]
        last_run = max(run_ids)
        path = os.path.join(self.run_dir, str(last_run))
        if not os.listdir(path):
            last_run = last_run - 1
        return last_run

    def next_run(self) -> int:
        return self.last_run() + 1

    def last_model(self) -> int:
        return self.last_run()

    def this_model(self) -> int:
        return self.next_run()

    def next_run_path(self) -> str:
        " create a new folder and return "
        return os.path.join(self.run_dir, str(self.next_run()))

    def last_model_short(self) -> str:
        lmp = self.last_model_path()
        return "/".join(lmp.split('/')[-3:]) if lmp else "None"

    def last_model_path(self) -> str:
        " return existing model file "
        m = self.last_model()
        if m < 1:
            return None
        path = os.path.join(self.model_dir, str(m), MODEL_NAME)
        return path

    def this_model_path(self) -> str:
        " Return a new folder, and file path within that folder "
        path = os.path.join(self.model_dir, str(self.this_model()))
        if not os.path.exists(path):
            os.makedirs(path)
        model_path = os.path.join(path, MODEL_NAME)
        return model_path

    def get_offset(self) -> int:
        last_steps = os.path.join(self.run_dir, str(self.last_run()))
        if not os.path.exists(last_steps):
            return 0
        steps_path = os.path.join(last_steps, "[0-9]*")
        steps_logs = glob.glob(steps_path)
        if not len(steps_logs):
            return 0
        steps = [r.split('/')[-1:][0] for r in steps_logs]
        last_step = int(max(steps))
        next_step = last_step + 1
        return next_step

    def get_model_state(self) -> ModelState :
        ms = ModelState()
        ms.last_model_path = self.last_model_path()
        ms.last_model_short = self.last_model_short()
        ms.this_model_path = self.this_model_path()
        ms.next_run_path = self.next_run_path()
        ms.run = self.next_run()
        ms.step = self.get_offset()
        return ms

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

    def trainNetwork(self, iterations, samples, run, start, trainloader, trainset, testloader, optimizer, criterion, writer) -> int:
        """ loop over the dataset multiple times """
        print('Starting Training at run {} step {}'.format(run, start))
        step = start
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
                running_loss += loss.item()

                # print statistics every SAMPLE_SIZE mini-batches
                step = step + 1
                if i % samples == samples - 1 :
                    print('%d: [%d, %5d] loss: %.3f' % (run, epoch + 1, step, running_loss / samples))

                    # ...log the running loss
                    writer.add_scalar('training loss', running_loss / samples, step)

                    # ...log a Matplotlib Figure showing the model's predictions on a random mini-batch
                    # writer.add_figure('predictions vs. actuals',
                    #                   self.plot_classes_preds(inputs, labels),
                    #                   global_step=step)

                    writer.close()

                    self.logStepEmbedding(trainset, writer, step)

                    self.add_precision_recall(testloader, writer, step)

                    running_loss = 0.0

        print('Finished Training at run {} step {}'.format(run, step))
        return step

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

    def getState(self) :
        self.state = State()
        self.model = self.state.get_model_state()

    def clearState(self):
        """
        Clear any persistent state, and reinitialise from a blank state
        :return:
        """
        self.state.clear()
        self.getState()
        self.setNetwork()

    def setWriter(self):
        path = self.model.next_run_path
        if not os.path.exists(path):
            os.makedirs(path)
        self.writer = SummaryWriter(path)

    def setOptimizer(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.001, momentum=0.9)

    def setNetwork(self):
        if self.model.last_model_path and os.path.exists(self.model.last_model_path):
            self.loadNetwork()
        else:
            print("Starting Model afresh..")
            self.state.clear()
            self.getState()
            self.network = CnvNet()
        self.setOptimizer()

    def loadNetwork(self):
        cnvnet = CnvNet()
        cnvnet.load_state_dict(torch.load(self.model.last_model_path))
        print( "Loading Model State from {} at step {}.".format(self.model.last_model_short, self.model.step))
        self.network = cnvnet

    def saveNetwork(self):
        this_model = self.model.this_model_path
        state = self.network.state_dict()
        print("Writing Model State to {}".format(this_model))
        torch.save(state, this_model)

    def setImageData(self):
        self.imgData = ImageData()
        self.imgData.downloadImages()

    def __init__(self):
        self.getState()
        print("Starting Run at {}, ({},{})".format(self.model.next_run_path, self.model.run, self.model.step))
        self.setNetwork()

        self.setImageData()

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

    def trainAndReport(self, iterations=1, samples=2000):
        # self.logSampleImages()
        self.setWriter()
        self.network.trainNetwork(iterations,
                                  samples,
                                  self.model.run,
                                  self.model.step,
                                  self.imgData.trainloader, self.imgData.trainset,
                                  self.imgData.testloader,
                                  self.optimizer, self.criterion,
                                  self.writer)
        self.getState()
        self.saveNetwork()
        self.network.testNetworkSet(self.imgData.testloader)
        self.network.classAccuracy(self.imgData.testloader)

