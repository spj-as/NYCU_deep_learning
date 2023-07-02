from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader
import torch
from torchvision.models import resnet50
from tqdm import tqdm
from plot import plot_acc, plot_confusion
import torch
import random
import numpy as np
import argparse

# Seed
seed = 13
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def getParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=8)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser


def train(train_loader, test_loader, epochs, model, loss_function, optimizer, mode):
    best_acc = 0.0
    if mode: name = "pre"
    else: name = ""
    train_acc = []
    test_acc = [] 
    total = 0.0
    for epoch in range(epochs):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        correct = 0.0
        total = 0.0
        total_loss = 0.0
        for i, data in enumerate(tqdm(train_loader)):
           
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            total += labels.size(0)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum()
            accuracy = 100. * correct / total
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            
            if i % 100 == 0:
                print('\n[epoch:%d, iter:%d] Loss: %.05f | Acc: %.5f%% ' 
                    % (epoch + 1, i, total_loss /(i+1), accuracy))
            # if best_acc < accuracy:
            #     torch.save(model.state_dict(), name+'_resnet50best.pt')
            #     best_acc = accuracy
        acc = 100. * correct / total
        train_acc.append(acc.tolist())
             
        total = 0
        correct = 0
        predict_ = []
        label_ = []
        with torch.set_grad_enabled(False):
            for data in (tqdm(test_loader)):
                model.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
                for i in predicted.tolist():
                    predict_.append(i)
                for i in labels.tolist():
                    label_.append(i)
    
        acc = 100 * correct /  total
        if best_acc < acc:
            torch.save(model.state_dict(), name+'_resnet50best.pt')
            best_acc = acc
        print('Test\'s accuracy is: %.3f%%' % (acc))
        test_acc.append(acc.tolist())  
    
    with open('ResNet50'+name+'_train_acc.txt', 'w') as f:
        for item in train_acc:
            f.write(str(item)+"\n")
    with open('ResNet50'+name+'_test_acc.txt', 'w') as f:
        for item in test_acc:
            f.write(str(item)+"\n")    
    plot_confusion(label_, predict_,"ResNet50")

class ResNet50(torch.nn.Module):
    def __init__(self,numclass, pretrained):

        super(ResNet50,self).__init__()
        self.model=resnet50(pretrained=pretrained)
        self.model.fc = torch.nn.Linear(in_features=self.model.fc.in_features, out_features=numclass)

        # torch.nn.Sequential(
        #         torch.nn.Flatten(),
        #         torch.nn.Linear(in_features=self.model.fc.in_features, out_features=100),
        #         torch.nn.ReLU(inplace=True),
        #         torch.nn.Dropout(p=0.3),
        #         torch.nn.Linear(in_features=100, out_features=5),
        #                    )
        
    def forward(self, X):
        out = self.model(X)
        return out

if __name__ == '__main__':
    args = getParser()
    args = args.parse_args()
    epochs = args.epoch
    batch_size = args.batch_size
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    train_dataset = RetinopathyLoader("./data/new_train", "train")
    test_dataset = RetinopathyLoader("./data/new_test", "test")
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=8)
    device = ('cuda' if torch.cuda.is_available else 'cpu')
    
    pretrain = [False, True]
    for mode in pretrain:
        model = ResNet50(numclass = 5, pretrained = mode)
        model = model.to(device)
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=momentum, weight_decay=weight_decay)
        print(device)
        # model.load_state_dict(torch.load("resnet_50best.pt")) 
        train(train_loader, test_loader, epochs, model, loss_function, optimizer, mode)
    plot_acc(epochs, 'ResNet50')
