import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import F1Score, MeanSquaredError
from torchvision.ops import sigmoid_focal_loss


class CNN_face_and_age_model(pl.LightningModule):
    def __init__(self, layers, n, dr, fl, lr):
        super().__init__()
        self.save_hyperparameters()
        
        self.layers = layers
        self.n = n
        self.dr = dr
        self.lr = lr
        
        self.gender_f1_train = F1Score(task='binary')
        self.gender_f1_val = F1Score(task='binary')
        
        self.age_rmse_train = MeanSquaredError(squared=False)
        self.age_rmse_val = MeanSquaredError(squared=False)
        
        self.gender_loss = nn.BCEWithLogitsLoss()
        self.age_loss =  nn.SmoothL1Loss(beta=0.1)
        
        CNN_models = []
        self.prev_channel = 3

        for i in range(self.layers):           
            out_channels = min(fl * (2**i), 512)             
            CNN_models.extend([
                nn.Conv2d(  in_channels=self.prev_channel, 
                            out_channels=out_channels, 
                            kernel_size=self.n, 
                            padding=1,
                            stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout(self.dr),

            ])
            self.prev_channel = out_channels
        
        self.CNN_model = nn.Sequential(*CNN_models)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.age_head = nn.Sequential(
            nn.Linear(self.prev_channel, 256),
            nn.BatchNorm1d(256), 
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.gender_head = nn.Sequential(
            nn.Linear(self.prev_channel, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dr),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, mode="max", factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "final_score", 
            },
        }

    
    def training_step(self, batch, batch_idx):
        x, y_gender, y_age = batch
        
        y_gender = y_gender.view(-1, 1).float()
        y_age = y_age.view(-1, 1).float()

        gender_logits, age_preds = self(x)
        
        loss_g = sigmoid_focal_loss(
            gender_logits, 
            y_gender, 
            alpha=0.248, 
            gamma=2.0, 
            reduction='mean'
        )
        loss_a = self.age_loss(age_preds, y_age)

        loss = loss_g + loss_a*2

        gender_preds = (torch.sigmoid(gender_logits) > 0.5).int()       
        self.gender_f1_train(gender_preds, y_gender.int())
        self.age_rmse_train(age_preds*100, y_age*100)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_gender_loss', loss_g, on_epoch=True)
        self.log('train_age_loss', loss_a, on_epoch=True)
        self.log('train_gender_f1', self.gender_f1_train, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_age_rmse', self.age_rmse_train, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y_gender, y_age = batch
        
        y_gender = y_gender.view(-1, 1).float()
        y_age = y_age.view(-1, 1).float()

        gender_logits, age_preds = self(x)
        
        loss_g = sigmoid_focal_loss(
            gender_logits, 
            y_gender, 
            alpha=0.248, 
            gamma=2.0, 
            reduction='mean'
        )
        loss_a = self.age_loss(age_preds, y_age)
        
        loss = loss_g + loss_a*2

        gender_preds = (torch.sigmoid(gender_logits) > 0.5).int()
        
        self.gender_f1_val(gender_preds, y_gender.int())
        self.age_rmse_val(age_preds*100, y_age*100)

        probs = torch.sigmoid(gender_logits)
        self.log('gender_prob_mean', probs.mean())
        self.log('gender_prob_std', probs.std())

        self.log("val_gender_loss" , loss_g, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_age_loss", loss_a, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_gender_f1', self.gender_f1_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_age_rmse', self.age_rmse_val, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        epoch_age_rmse = self.age_rmse_val.compute()
        epoch_gender_f1 = self.gender_f1_val.compute()
        
        age_score = 1.0 - (torch.clamp(epoch_age_rmse, max=30.0) / 30.0)
        final_score = (2 * age_score * epoch_gender_f1) / (age_score + epoch_gender_f1 + 1e-6)
        self.log('final_score', final_score, prog_bar=True)

    def forward(self, x):
        x = self.CNN_model(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        gender = self.gender_head(x)
        age = self.age_head(x)
        return gender, age
