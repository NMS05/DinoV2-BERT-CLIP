import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.image_caption_data import CocoDataset
from model.clip import clip_dinov2_bert, ClipLoss

import time
import numpy as np


def train_one_epoch(train_data_loader, model, optimizer, loss_fn, device):
    
    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0
    
    model.train()

    ###Iterating over data loader
    for i, (images, input_ids, attention_mask) in enumerate(train_data_loader):
        
        #Loading data and labels to device
        images = images.to(device)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        #Reseting Gradients
        optimizer.zero_grad()
        #Forward
        dino_features, bert_features = model(images, input_ids, attention_mask)
        #Calculating Loss
        _loss = loss_fn(dino_features, bert_features)
        epoch_loss.append(_loss.item())      
        #Backward
        _loss.backward()
        optimizer.step()

        # calculate acc per minibatch
        logits,_ = loss_fn.get_logits(dino_features, bert_features)
        labels = loss_fn.get_ground_truth(dino_features.device, dino_features.shape[0])
        sum_correct_pred += (torch.argmax(logits,dim=-1) == labels).sum().item()
        total_samples += len(labels)
    
        if i%10 == 0: print("train_loss = ",_loss.item())

    acc = round(sum_correct_pred/total_samples,4)*100
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    return epoch_loss, acc

def val_one_epoch(val_data_loader, model, loss_fn, device):
    
    ### Local Parameters
    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0

    model.eval()

    with torch.no_grad():
        ###Iterating over data loader
        for images, input_ids, attention_mask in val_data_loader:
            
            #Loading data and labels to device
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            #Forward
            dino_features, bert_features = model(images, input_ids, attention_mask)
            #Calculating Loss
            _loss = loss_fn(dino_features, bert_features)
            epoch_loss.append(_loss.item())
            
            # calculate acc per minibatch
            logits,_ = loss_fn.get_logits(dino_features, bert_features)
            labels = loss_fn.get_ground_truth(dino_features.device, dino_features.shape[0])
            sum_correct_pred += (torch.argmax(logits,dim=-1) == labels).sum().item()
            total_samples += len(labels)

    acc = round(sum_correct_pred/total_samples,4)*100
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    return epoch_loss, acc


def train_clip(batch_size, epochs):
    """
    DataLoader
    """
    # Define the paths to the dataset and annotations
    train_dir = "MSCOCO/train2017/"
    train_anno = "MSCOCO/annotations/captions_train2017.json"
    val_dir = "MSCOCO/val2017/"
    val_anno = "MSCOCO/annotations/captions_val2017.json"
    # Create the dataset and dataloader
    train_dataset = CocoDataset(train_dir, train_anno, apply_transform=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    val_dataset = CocoDataset(val_dir, val_anno, apply_transform=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
  
    """
    Model and Loss
    """
    model = clip_dinov2_bert()
    device = torch.device("cuda:0")
    model = nn.DataParallel(model,device_ids=[0,1,2,3])
    model.to(device)
    print("\n\n\n\n\t Model Loaded")
    print("\t Total Params = ",sum(p.numel() for p in model.parameters()))
    print("\t Trainable Params = ",sum(p.numel() for p in model.parameters() if p.requires_grad))

    """
    Train
    """
    loss_fn = ClipLoss(temperature=0.1)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    print("\n\t Started Training\n")

    for epoch in range(epochs):

        begin = time.time()

        ###Training
        loss, acc = train_one_epoch(train_loader, model, optimizer, loss_fn, device)
        ###Validation
        val_loss, val_acc = val_one_epoch(val_loader, model, loss_fn, device)

        print('\n\n\t Epoch....', epoch + 1)
        print("\t Training loss & accuracy......",round(loss,4), round(acc,2))
        print("\t Val loss & accuracy......", round(val_loss,4), round(val_acc,2))
        print('\t Time per epoch (in mins) = ', round((time.time()-begin)/60,2),'\n\n')

    torch.save(model.state_dict(),'clip.pth')

if __name__=="__main__":
    train_clip(batch_size=512, epochs=5)