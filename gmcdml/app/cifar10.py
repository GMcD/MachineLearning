import torch
import torchvision
import torchvision.transforms as transforms
from gmcdml.app.utils import imshow

class ImageData(object):

    trainset = None
    trainloader = None

    testset = None
    testloader = None

    classes = ()

    def downloadImages(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=4, shuffle=True, num_workers=2)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=4, shuffle=False, num_workers=2)


    def showImages(self):
        # get some random training images
        dataiter = iter(self.trainloader)
        images, labels = dataiter.next()

        # show images
        imshow(torchvision.utils.make_grid(images))
        # print labels
        print(' '.join('%5s' % self.classes[labels[j]] for j in range(4)))

def downloadAndShow():
    id = ImageData()
    id.downloadImages()
    id.showImages()
