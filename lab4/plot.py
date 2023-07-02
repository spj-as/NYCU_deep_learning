import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_acc(epoch, model):
    epochs = [i for i in range (epoch)]
    plt.figure()
    with open(model+"pre_train_acc.txt", "r") as file: 
        line = file.read().splitlines()
    pretrain_train_acc =[float(l) for l in line]
    with open(model+"pre_test_acc.txt", "r") as file: 
        line = file.read().splitlines()
    pretrain_test_acc =[float(l) for l in line]
    with open(model+"_test_acc.txt", "r") as file: 
        line = file.read().splitlines()
    test_acc =[float(l) for l in line]
    with open(model+"_train_acc.txt", "r") as file: 
        line = file.read().splitlines()
    train_acc =[float(l) for l in line]
    plt.plot(epochs, test_acc, label='Test(w/o pretraining)')
    plt.plot(epochs, pretrain_test_acc, label='Test(with pretraining)')
    plt.plot(epochs, train_acc, label='Train(w/o pretraining)')
    plt.plot(epochs, pretrain_train_acc, label='Train(with pretraining)')

    plt.legend(loc='upper right')
    plt.ylabel('Accuracy %')  
    plt.xlabel('Epochs')  
    plt.legend()    
    plt.savefig('Result.png')

def plot_confusion(ground_truth, prediction, model):
    matrix = confusion_matrix(
        ground_truth,
        prediction,
        labels= [0, 1, 2, 3, 4],
        normalize='true',
    )
    disp = ConfusionMatrixDisplay(confusion_matrix = matrix,
                            display_labels = [0, 1, 2, 3, 4],)
    disp.plot(cmap=plt.cm.Blues)

    plt.title('Normalized confusion matrix')         
    plt.xlabel('Predicted label')        
    plt.ylabel('True label')
    plt.savefig(model + ' Normalized confusion matrix.png')
plot_acc(8, "Resnet50")    