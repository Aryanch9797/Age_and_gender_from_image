import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os

class FaceData(Dataset):
    """ Custom Dataset class for Training and Validation. """
    
    def __init__(self, df, is_val, data_path="/kaggle/input/sep-25-dl-gen-ai-nppe-1/face_dataset/"):
        self.df = df
        self.data_path = data_path
        self.is_val = is_val
        
        val_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
        
        train_transforms = [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),    
            transforms.RandomVerticalFlip(p=0.),
            transforms.RandomRotation(degrees=15),            
            transforms.ColorJitter(brightness=0.2,           
                                   contrast=0.2, 
                                   saturation=0.2, 
                                   hue=0.05),
            transforms.ToTensor(),
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
        mean = img_tensor.mean()
        std = img_tensor.std()
        img_tensor_scaled = (img_tensor - mean) / (std + 1e-7) 
        
        return img_tensor_scaled, torch.tensor(gender, dtype=torch.float32), torch.tensor(age, dtype=torch.float32)
