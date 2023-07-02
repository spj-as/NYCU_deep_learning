import pandas as pd
import numpy as np
from torch.utils import data
from PIL import Image, ImageFile
import os
import torchvision.transforms as transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True

def transformer(mode, img):
    if mode == "train":
        transformer = transforms.Compose([ 
                                    transforms.RandomRotation(180),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.Resize([512,512]),
                                    transforms.CenterCrop(512, padding=16, padding_mode='reflect'),
                                    transforms.ToTensor(),
                                  ])
        
        return(transformer(img))
    else:
        transformer = transforms.Compose([ 
                                        transforms.Resize([512,512]),
                                        transforms.ToTensor(),
                                        ])
        return(transformer(img))
def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv', header=None)
        label = pd.read_csv('train_label.csv', header=None)
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv', header=None)
        label = pd.read_csv('test_label.csv', header=None)
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        
        img = Image.open(os.path.join(self.root, self.img_name[index]) + '.jpeg')
        label = self.label[index]
        img = transformer(self.mode, img)

        return img, label
