

"""
Trains classifier
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch
import argparse
import numpy as np
import glob
import os
import time
from sklearn import metrics
import json
import datetime
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import sklearn.metrics as metrics
import wandb


NUM_CLASSES = 9


class CRC_Feat_Dataset:
    def __init__(self, root_dir):
        self.root_dir=root_dir
        self.files=sorted(glob.glob(os.path.join(root_dir, '*', '*.npy')))
        self.targets = [file.split('/')[-2] for file in self.files]
        self.class2id = {j:i for i, j in enumerate(sorted(os.listdir(root_dir)))}
        self.targets = [self.class2id[t] for t in self.targets]

    def __getitem__(self, index):
        file = self.files[index]
        target = self.targets[index]
        return torch.Tensor(np.load(file)), target, file

    def __len__(self):
        return len(self.files)


def update_stats_json(json_file, epoch, loss_trn, acc_trn, loss_val, acc_val, report):
    if os.path.isfile(json_file)
        dict = json.load(json_file)
    else:
        dict = []
    curr_dict = {"epoch":epoch, "loss_trn":loss_trn, "acc_trn":acc_trn, "loss_val":loss_val, "acc_val":acc_val, "report":report}
    dict.append({curr_dict})
    fh = open(json_file, 'w')
    json.dump(dict, fh, indent=4)
    fh.close()


def train_epoch(model, dataloader, optimizer, device, writer, epoch):
    running_correct = 0
    running_loss = 0
    criterion = nn.CrossEntropyLoss()

    for i, (data,target,_) in enumerate(dataloader):
        data,target = data.to(device),target.to(device)
        out = model(batch)

        optimizer.zero_grad()
        loss = criterion(input=output,target=target)
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        running_correct += correct
        running_loss += loss.item()*len(data)
        acc_batch = correct/len(data)

        writer.add_scalar("Loss/Train", loss.item(), epoch*len(dataloader)+i)
        writer.add_scalar("Acc/Train", acc_batch, epoch*len(dataloader)+i)

    epoch_loss = running_loss/len(dataloader.dataset)
    epoch_acc = running_correct/len(dataloader.dataset)
    return epoch_loss, epoch_acc


def val_epoch(model, dataloader, device, writer, epoch):
    y_pred = []
    y_gt = []
    criterion = nn.CrossEntropyLoss()

    for i, (data,target,_) in enumerate(dataloader):
        data,target = data.to(device),target.to(device)
        out = model(batch)

        loss = criterion(input=output,target=target)

        pred = out.argmax(dim=1, keepdim=True)
        running_loss += loss.item()*len(data)
        y_gt.append(target.tolist())
        y_pred.append(pred.tolist())

    epoch_loss = running_loss/len(dataloader.dataset)
    epoch_acc = running_correct/len(dataloader.dataset)
    report = metrics.classification_report(y_gt, y_pred, digits=4, output_dict=True, \
                labels=range(NUM_CLASSES), target_names=sorted(os.listdir(dataloader.dataset.dataset.root_dir)))
    return epoch_loss, epoch_acc, report


def train_classification_model(root_dir, checkpoint, batch_size, ckpt_dir, log_dir, num_epochs, learning_rate):
    if checkpoint is not None:
        assert os.path.isfile(checkpoint, "Provided checkpoint not found.")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    dataset = CRC_Feat_Dataset(root_dir=root_dir)

    train_idx,val_idx = train_test_split(range(len(dataset)), test_size=.2, stratify=dataset.targets, random_state=42)
    train_dset = Subset(dataset, train_idx)
    val_dset = Subset(dataset, val_idx)

    train_dl = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dset, batch_size=batch_size, shuffle=False)

    resnet = models.resnet18(pretrained = True)
    resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)
    layers = [resnet.layer4[1], resnet.avgpool, nn.Flatten(), resnet.fc]
    model = nn.Sequential(*layers)

    state_dict = torch.load()
    if checkpoint is not None:
        model.load_state_dict(checkpoint)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model).to(device)

    wandb.init(project='ILAWSIA', sync_tensorboard=True)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        model.train()
        loss_trn, acc_trn = train_epoch(model, train_dl, optimizer, device, log_dir, writer, epoch)
        ckpt_fp = os.path.join(ckpt_dir, "classifier_ep{}.pt".format(epoch))
        torch.save(model.module.state_dict(), ckpt_fp)
        model.eval()
        with torch.no_grad():
            loss_val, acc_val, report = val_epoch(model, val_dl, device, log_dir, writer, epoch)
        stats_file = os.path.join(log_dir, "stats.json")
        update_stats_json(stats_file, epoch, loss_trn, acc_trn, loss_val, acc_val, report)
    wandb.finish()


def main():
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    parser = argparse.ArgumentParser(description='Measures peformance against test set')
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--log_dir", type=int, default=os.path.join("logs", time))
    args = parser.parse_args()
    print(args,flush=True)
    train_classification_model(args.root_dir, args.checkpoint, args.batch_size, \
            args.ckpt_dir, args.log_dir, args.num_epochs, args.learning_rate)



if __name__=="__main__":
    main()