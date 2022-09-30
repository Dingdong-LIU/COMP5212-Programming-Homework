import numpy as np
from torch import optim
from models import Logistic_Regression, logistic_loss, svm_loss
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn as nn
import torch
from torch.utils.data.sampler import SubsetRandomSampler

# Don't change batch size
batch_size = 64

def load_minist():
    train_data = datasets.MNIST('~/data/mnist', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
    test_data = datasets.MNIST('~/data/mnist', train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
    subset_indices = ((train_data.train_labels == 0) +
                    (train_data.train_labels == 1)).nonzero().squeeze(1)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=False, sampler=SubsetRandomSampler(subset_indices))


    subset_indices = ((test_data.test_labels == 0) +
                    (test_data.test_labels == 1)).nonzero().squeeze(1)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, sampler=SubsetRandomSampler(subset_indices))
    return train_loader, test_loader


# Get a batch from training set and do some experiments
for i, (images, labels) in enumerate(train_loader):
    images = Variable(images.view(-1, 28*28))
    # Convert labels from 0,1 to -1,1
    labels = Variable(2*(labels.float()-0.5))
    break
images.shape, labels.shape


# Training the Model
# Notice that newest Pytorch merge tensor and variable, so the additional Variable wrapping is no longer required.

model = Logistic_Regression(28*28, 1)
loss_fn = svm_loss
print(loss_fn(model(images), labels))

num_epochs = 20  # added by me. modifiable.
opt = optim.SGD(model.parameters(), lr=5e-3)
device = torch.device(
    'cpu') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for epoch in range(num_epochs):
    total_loss = []
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28*28)).to(device)
        labels = Variable(2*(labels.float()-0.5)).to(device)

        pred = model(images)
        loss = loss_fn(pred, labels)

        loss.backward()
        opt.step()
        opt.zero_grad()

        total_loss.append(loss.cpu().detach().numpy())
        total_loss = np.mean(total_loss)
        print(f"Epoch {epoch}, loss = {total_loss}")


# Test the Model
correct = 0.
total = 0.
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28)).to(device)

    pred = model(images).squeeze()
    prediction = torch.sigmoid(pred).round().cpu().detach()

    correct += (prediction.view(-1).long() == labels).sum()
    total += images.shape[0]
print('Accuracy of the model on the test images: %f %%' %
      (100 * (correct.float() / total)))
