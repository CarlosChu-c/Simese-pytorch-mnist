import torch
import torchvision.datasets as dset
import torchvision.utils
from torch import optim
from torch.utils.data import DataLoader

from dataset import getDataset
from helpFunction import imshow, show_plot
from loss import ContrastiveLoss
from net import SiameseNetwork


# Config
class Config:
    training_dir = "./data/training/"
    train_batch_size = 64
    train_number_epochs = 50


# Load Train Dataset
train_dataset = dset.MNIST(root=Config.training_dir, train=True)
siamese_dataset = getDataset(train_dataset)

# Show Dataset Example
vis_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            num_workers=8,
                            batch_size=8)
dataiter = iter(vis_dataloader)
example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy())
train_dataloader = DataLoader(siamese_dataset,
                              shuffle=True,
                              num_workers=8,
                              batch_size=Config.train_batch_size)

# Start Training
net = SiameseNetwork().cuda()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005)
counter = []
loss_history = []
iteration_number = 0

# Training
for epoch in range(0, Config.train_number_epochs):
    for i, data in enumerate(train_dataloader, 0):
        img0, img1, label = data
        img0, img1, label = img0.type(torch.FloatTensor), img1.type(torch.FloatTensor), label.type(torch.FloatTensor)
        img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
        optimizer.zero_grad()
        output1, output2 = net(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        if i % 10 == 0:
            print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
show_plot(counter, loss_history)

# Save Model
torch.save(net, './model')
