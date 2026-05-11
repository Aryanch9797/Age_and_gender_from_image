""" Custom Dataset class for Training and Validation. """

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class FaceData(Dataset):
    
    def __init__(self, df, is_val, data_path="/kaggle/input/sep-25-dl-gen-ai-nppe-1/face_dataset/"):
        self.df = df
        self.data_path = data_path
        self.is_val = is_val

        mean_ds = [0.477, 0.412, 0.380] 
        std_ds  = [0.292, 0.273, 0.272] 
        
        val_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_ds, std=std_ds)
        ]

  

        train_transforms = [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),    
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=10),            
            transforms.ColorJitter(brightness=0.2,           
                                   contrast=0.2, 
                                   saturation=0.2, 
                                   hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_ds, std=std_ds)
        ]
        
        if self.is_val:
            self.transform = transforms.Compose(val_transforms)
        else:
            self.transform = transforms.Compose(train_transforms)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["full_path"]
        gender = row["gender"]
        age = row["age"]
        
        full_img_path = os.path.join(self.data_path, img_path)
        img = Image.open(full_img_path).convert('RGB')
        
        img_tensor = self.transform(img)
        
        return img_tensor, torch.tensor(gender, dtype=torch.float32), torch.tensor(age/100, dtype=torch.float32)
