import os 
import random
import cv2
import torch
from torchvision import transforms 
from PIL import Image
import numpy as np

class DataLoader:
    def __init__(self, path , shape):
        self.shape = shape
        h,w,c = shape
        with open(path, 'r') as f:
            self.trainlist = f.read().splitlines()
        self.transforms = transforms.Compose([
            transforms.RandomCrop(h),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __call__(self):
        h,w,c = self.shape
        self.trainlist = random.sample(self.trainlist, len(self.trainlist))

        for sample in self.trainlist:
            img1_path = sample.split(',')[0]
            img2_path = sample.split(',')[1]
            img3_path = sample.split(',')[2]

            ftminus1 = Image.open(img1_path)
            ftminus1 = ftminus1.resize((h,w),Image.BILINEAR)
            ft = Image.open(img2_path)
            ft = ft.resize((h,w),Image.BILINEAR)
            ftplus1 = Image.open(img3_path)
            ftplus1 = ftplus1.resize((h,w),Image.BILINEAR)
            fs = random_translation(ft)
    
            #augmentation
            seed = random.randint(0,2**32)
            torch.manual_seed(seed)
            ftminus1 = self.transforms(ftminus1)
            torch.manual_seed(seed)
            ft = self.transforms(ft)
            torch.manual_seed(seed)
            fs = self.transforms(fs)
            torch.manual_seed(seed)
            ftplus1 = self.transforms(ftplus1)

            yield ftminus1, ft, fs, ftplus1


def random_translation(img):
    img = np.array(img)
    (h,w) = img.shape[:-1]
    dx = np.random.randint(-w//10,w//10)
    dy = np.random.randint(-h//10,h//10)
    mat = np.array([[1,0,dx],[0,1,dy]],dtype=np.float32)
    img = cv2.warpAffine(img, mat, (w,h))
    return Image.fromarray(img)