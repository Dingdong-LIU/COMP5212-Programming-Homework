import numpy as np
from torch import optim
from models import Linear_Layer, logistic_loss, svm_loss
from torchvision import datasets, transforms
from torch.autograd import Variable
# import torch.nn as nn
import torch
from torch.utils.data.sampler import SubsetRandomSampler


# Don't change batch size
def load_minist(batch_size=64):
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


def load_model():
    model = Linear_Layer(28*28, 1)
    return model

def get_loss_fn(loss_fn_name):
    loss_fn_name = loss_fn_name.lower()
    assert loss_fn_name in ['logistic regression', 'svm'], f"{loss_fn_name} is not implemented"
    if loss_fn_name == "logistic regression":
        return logistic_loss
    if loss_fn_name == 'svm':
        return svm_loss
    return None


def train_and_test_model(loss_type="logistic regression", momentum=0, step_size_set=None):
    train_loader, test_loader = load_minist()
    model = load_model()

    # get loss function
    loss_fn = get_loss_fn(loss_type)

    # get learning rate
    if not step_size_set:
        step_size_set = [5e-3]

    train_test_records = {
        'model_type' : loss_type,
        'train_history' : []
    }
    for lr in step_size_set:
        # get optimizer
        opt = optim.SGD(model.parameters(), momentum=momentum, lr=lr)

        num_epochs = 20  # added by me. modifiable.
        device = torch.device(
            'cpu') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        # Train the Model
        loss_history = []
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
            loss_history.append(total_loss)
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
        accuracy = (100 * (correct.float() / total))
        print('Accuracy of the model on the test images: %f %%' % accuracy)

        train_test_records['train_history'].append({
            'lr' : lr,
            'test_acc' : accuracy,
            'training_history' : loss_history,
        })
    return train_test_records

if __name__ == "__main__":
    train_and_test_model()
