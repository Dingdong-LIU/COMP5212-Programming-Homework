import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
# import torchvision.transforms as transforms
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import random
import time
import matplotlib.pyplot as plt
from models import MLP, CNN, cnn_preprocessing, mlp_preprocessing
from tqdm.auto import tqdm

def cifar_loaders(batch_size, shuffle_test=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('~/data/cifar10', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomCrop(32, 4),
                                 transforms.ToTensor(),
                                 normalize,
                             ]))
    test = datasets.CIFAR10('~/data/cifar10', train=False,
                            transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                              shuffle=shuffle_test, pin_memory=True)
    return train_loader, test_loader

def train_model(model, device, train_loader, num_epoch, opt, loss_fn, preprocessing_fn):
    model.to(device)
    epoch_loss_history = []
    for epoch in range(num_epoch):
        loss_history = []
        correct = 0
        with tqdm(train_loader) as progress_bar:
            progress_bar.set_description(f'Train epoch {epoch}')
            for i, (images, labels) in enumerate(progress_bar):
                num_pictures = images.shape[0]
                # images = Variable(images.view(-1, 3*32*32)).to(device)
                # labels = labels.to(device)
                images, labels = preprocessing_fn(images, labels)
                images, labels = images.to(device), labels.to(device)
                ## Prediction
                output = model(images)
                loss = loss_fn(output, labels)

                with torch.no_grad():
                    predictions = output.argmax(dim=1).squeeze()
                    correct_in_batch = (predictions == labels).sum().item()
                    accuracy = correct_in_batch / num_pictures
                    correct += correct_in_batch

                loss.backward()
                opt.step()
                opt.zero_grad()

                loss_history.append(loss.cpu().detach().numpy())
                # current_total_loss = np.mean(loss_history)
                progress_bar.set_postfix(batch_loss=loss.item(), accuracy=accuracy)
                if i == len(train_loader) - 1:
                    progress_bar.set_postfix(
                        Accuracy=correct/len(train_loader.dataset), Total_loss=np.mean(loss_history))
            epoch_loss_history.append(np.mean(loss_history))
    return epoch_loss_history, correct/len(train_loader.dataset)

def test_model(model, device, test_loader, preprocessing_fn):
    model.to(device)
    # loss_history = []
    correct = 0
    with tqdm(test_loader) as progress_bar:
        progress_bar.set_description(f'Testing')
        for i, (images, labels) in enumerate(progress_bar):
            num_pictures = images.shape[0]
            # images = Variable(images.view(-1, 3*32*32)).to(device)
            # labels = labels.to(device)
            with torch.no_grad():
                images, labels = preprocessing_fn(images, labels)
                images, labels = images.to(device), labels.to(device)
                ## Prediction
                output = model(images)

                predictions = output.argmax(dim=1).squeeze()
                correct_in_batch = (predictions == labels).sum().item()
                batch_accuracy = correct_in_batch / num_pictures
                correct += correct_in_batch

                # loss_history.append(loss.cpu().detach().numpy())
            # current_total_loss = np.mean(loss_history)
            progress_bar.set_postfix(accuracy=batch_accuracy)
            if i == len(test_loader) - 1:
                progress_bar.set_postfix(
                    Accuracy=correct/len(test_loader.dataset))
    return model, correct/len(test_loader.dataset)

def main():
    # Load dataset 
    batch_size = 64
    test_batch_size = 64
    train_loader, _ = cifar_loaders(batch_size)
    _, test_loader = cifar_loaders(test_batch_size)

    # Load model
    model = MLP(3*32*32, 10)
    preprocessing_fn = mlp_preprocessing
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Get device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    train_loss_history, train_accuracy = train_model(model, device=device, train_loader=train_loader, num_epoch=3, opt=opt, loss_fn=loss_fn, preprocessing_fn=preprocessing_fn)

    model, test_accuracy = test_model(model, device=device, test_loader=test_loader, preprocessing_fn=preprocessing_fn)

    print(f"Train_acc = {train_accuracy}, Test_acc = {test_accuracy}")

# Default behavior
if __name__ == "__main__":
    main()
