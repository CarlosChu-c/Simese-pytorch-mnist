import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import getDataset
from helpFunction import plot_mnist
from helpFunction import imshow
from net import SiameseNetwork


# Config
class Config:
    testing_dir = "./data/testing/"


# Load Model
net = torch.load('./model')

# Load Test Dataset
dataset_test = dset.MNIST(root=Config.testing_dir, train=False)
siamese_dataset = getDataset(dataset_test, relables=True)
test_dataloader = DataLoader(siamese_dataset, num_workers=6, batch_size=1, shuffle=True)
dataiter = iter(test_dataloader)
numpy_all = []
numpy_labels = []
correct_pre = 0

# Testing
for i in range(10000):
    x0, x1, label2, label0, label1 = next(dataiter)
    output1, output2 = net(Variable(x0).type(torch.FloatTensor).cuda(), Variable(x1).type(torch.FloatTensor).cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    # Set the threshold value as 1.15 to identify if two images are similar
    if label2 == 0 and euclidean_distance.cpu()[0].detach().numpy() < 1.15:
        correct_pre += 1
    if label2 == 1 and euclidean_distance.cpu()[0].detach().numpy() >= 1.15:
        correct_pre += 1
    accuracy = correct_pre / (i + 1) * 100
    numpy_all.append(output1.cpu()[0].detach().numpy())
    numpy_labels.append(label0.numpy()[0])
    if (i + 1) % 100 == 0:
        print("Progress {}/10000\n Current accuracy {}%\n".format(i + 1, accuracy))
numpy_all = np.array(numpy_all)
numpy_labels = np.array(numpy_labels)

# Show data's distribution
plot_mnist(numpy_all, numpy_labels)

# Show samples
test_dataloader_samples = DataLoader(siamese_dataset, num_workers=6, batch_size=1, shuffle=True)
dataiter_samples = iter(test_dataloader_samples)
x0, _, _, _, _ = next(dataiter_samples)
for i in range(20):
    _, x1, label2, _, _ = next(dataiter_samples)
    concatenated = torch.cat((x0, x1), 0)

    output1, output2 = net(Variable(x0).type(torch.FloatTensor).cuda(), Variable(x1).type(torch.FloatTensor).cuda())
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
