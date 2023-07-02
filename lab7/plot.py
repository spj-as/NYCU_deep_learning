import numpy as np
import matplotlib.pyplot as plt 

def plot(epoch, loss, acc):
    epochs = [i for i in range (epoch)]
    plt.figure()
    plt.plot(epochs, loss, label='Train loss')
    plt.plot(epochs, acc, label='Test accuracy')
    # plt.plot(epochs, Acc_emm, label='EMM test accuracy')
    plt.legend(loc='upper right')
    plt.ylabel('Loss / Accuracy')  
    plt.xlabel('Epochs')  
    plt.legend()    
    plt.savefig('train_loss.png')