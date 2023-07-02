from dataloader import RetinopathyLoader
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from plot import plot_acc, plot_confusion
import torch
import random
import numpy as np
import argparse
from main_resnet18 import ResNet18
from main_resnet50 import ResNet50

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
    parser.add_argument("--model", type=str, default="Resnet50")
    parser.add_argument("--batch_size", type=int, default=16)
    return parser


def train(test_loader, model, mode):
    if mode: name = "pre"
    else: name = ""
  
             
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
        print('Test\'s accuracy is: %.3f%%' % (acc))
    
    plot_confusion(label_, predict_,"ResNet50")

if __name__ == '__main__':
    args = getParser()
    args = args.parse_args()
    batch_size = args.batch_size
    model= args.model
    test_dataset = RetinopathyLoader("./data/new_test", "test")
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=8)
    device = ('cuda' if torch.cuda.is_available else 'cpu')
    mode = True
    if model == "Resnet50":
        model = ResNet50(numclass = 5, pretrained = mode)
    else:
        model = ResNet18(numclass= 5, pretrained= mode)
            
    
    model = model.to(device)
    print(device)
    model.load_state_dict(torch.load("pre_resnet50best.pt")) 
    train(test_loader, model, mode)
