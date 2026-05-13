""" Custom dataloader for test samples """

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class facetest(Dataset):
    """ Custom class for testdataset  """
    
    def __init__(self,df,data_path="/kaggle/input/sep-25-dl-gen-ai-nppe-1/face_dataset/"):
        
        self.df = df
        self.data_path = data_path

        mean_ds = [0.477, 0.412, 0.380] 
        std_ds  = [0.292, 0.273, 0.272] 
        
        val_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_ds, std=std_ds)
        ]

    def __len__(self):

        return len(self.df)

    def __getitem__(self,idx):
        
        row = self.df.iloc[idx]
        path = row["full_path"]
        img = Image.open(self.data_path+path).convert('RGB')

        img_tensor = self.val_transforms(img)   

        return img_tensor
        
