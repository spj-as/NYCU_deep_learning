from dataloader import read_bci_data
from model import DeepConvNet
from plot import plot_acc
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import random
import numpy as np


# Seed
seed = 12345
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class createDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.label[idx])


train_data, train_label, test_data, test_label = read_bci_data()
train_dataset = createDataset(train_data, train_label)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=0)
test_dataset = createDataset(test_data, test_label)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

device = "cuda" if torch.cuda.is_available else "cpu"
lr = 5e-4
# DeepConvNet -> Relu
model = DeepConvNet(activation=nn.ReLU()).cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=2e-3)
model.to(device)
### Hyperparameters setting
epochs = 500
losses = []
Relu_train_acc = []
Relu_test_acc = []
Relu_train_max = 0
Relu_test_max = 0
print("DeepConvNet with Relu ")
for epoch in range(epochs):
    correct = 0
    for x, label in train_loader:
        model.train()
        x = x.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        out = model(x)
        loss = loss_function(out, label)
        optimizer.zero_grad()
        loss.backward()
        value, idx = out.max(dim=1)
        correct += (idx == label).sum()
        optimizer.step()
    acc = correct.item() / train_data.shape[0]
    if acc > Relu_train_max:
        Relu_train_max = acc
    print(
        "Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}%".format(
            epoch + 1, epochs, loss.item(), acc * 100
        )
    )
    Relu_train_acc.append(correct.item() / train_data.shape[0])

    test_correct = 0
    for x, label in test_loader:
        model.eval()
        x = x.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        out = model(x)
        loss = loss_function(out, label)
        value, idx = out.max(dim=1)
        test_correct += (idx == label).sum()
    acc = test_correct.item() / test_data.shape[0]
    if acc > Relu_test_max:
        Relu_test_max = acc
    print("Test: Loss: {:.3f}, Accuracy: {:.3f}%".format(loss.item(), acc * 100))
    Relu_test_acc.append(test_correct.item() / test_data.shape[0])

# DeepConvNet -> LeakyRelu
model = DeepConvNet(activation=nn.LeakyReLU()).cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=2e-2)
model.to(device)
### Hyperparameters setting
epochs = 500
losses = []
LeakyRelu_train_acc = []
LeakyRelu_test_acc = []
LeakyRelu_train_max = 0
LeakyRelu_test_max = 0
print("DeepConvNet with LeakyRelu ")
for epoch in range(epochs):
    correct = 0
    for x, label in train_loader:
        model.train()
        x = x.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        out = model(x)
        loss = loss_function(out, label)
        optimizer.zero_grad()
        loss.backward()
        value, idx = out.max(dim=1)
        correct += (idx == label).sum()
        optimizer.step()
    acc = correct.item() / train_data.shape[0]
    if acc > LeakyRelu_train_max:
        LeakyRelu_train_max = acc
    print(
        "Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}%".format(
            epoch + 1, epochs, loss.item(), acc * 100
        )
    )
    LeakyRelu_train_acc.append(correct.item() / train_data.shape[0])

    test_correct = 0
    for x, label in test_loader:
        model.eval()
        x = x.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        out = model(x)
        loss = loss_function(out, label)
        value, idx = out.max(dim=1)
        test_correct += (idx == label).sum()
    acc = test_correct.item() / test_data.shape[0]
    if acc > LeakyRelu_test_max:
        LeakyRelu_test_max = acc
    print("Test: Loss: {:.3f}, Accuracy: {:.3f}%".format(loss.item(), acc * 100))
    LeakyRelu_test_acc.append(test_correct.item() / test_data.shape[0])

# DeepConvNet -> ELU
model = DeepConvNet(activation=nn.ELU(0.8)).cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=2e-2)
model.to(device)
### Hyperparameters setting
epochs = 500
losses = []
ELU_train_acc = []
ELU_test_acc = []
ELU_train_max = 0
ELU_test_max = 0
print("DeepConvNet with ELU ")
for epoch in range(epochs):
    correct = 0
    for x, label in train_loader:
        model.train()
        x = x.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        out = model(x)
        loss = loss_function(out, label)
        optimizer.zero_grad()
        loss.backward()
        value, idx = out.max(dim=1)
        correct += (idx == label).sum()
        optimizer.step()
    acc = correct.item() / train_data.shape[0]
    if acc > ELU_train_max:
        ELU_train_max = acc
    print(
        "Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}%".format(
            epoch + 1, epochs, loss.item(), acc * 100
        )
    )
    ELU_train_acc.append(correct.item() / train_data.shape[0])

    test_correct = 0
    for x, label in test_loader:
        model.eval()
        x = x.to(device, dtype=torch.float)
        label = label.to(device, dtype=torch.long)
        out = model(x)
        loss = loss_function(out, label)
        value, idx = out.max(dim=1)
        test_correct += (idx == label).sum()
    acc = test_correct.item() / test_data.shape[0]
    if acc > ELU_test_max:
        ELU_test_max = acc
    print("Test: Loss: {:.3f}, Accuracy: {:.3f}%".format(loss.item(), acc * 100))
    ELU_test_acc.append(test_correct.item() / test_data.shape[0])


plot_acc(
    Relu_train_acc,
    Relu_test_acc,
    LeakyRelu_train_acc,
    LeakyRelu_test_acc,
    ELU_train_acc,
    ELU_test_acc,
)
print("Relu_train_max:", Relu_train_max)
print("Relu_test_max:", Relu_test_max)
print("LeakyRelu_train_max:", LeakyRelu_train_max)
print("LeakyRelu_test_max:", LeakyRelu_test_max)
print("ELU_train_max:", ELU_train_max)
print("ELU_test_max:", ELU_test_max)
print(max(Relu_test_max, LeakyRelu_test_max, ELU_test_max))