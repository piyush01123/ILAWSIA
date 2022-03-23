

"""
Trains metric learning model using triplet loss
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
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


def update_stats_json(json_file, epoch, loss_trn, loss_val, acc_metrics):
    if os.path.isfile(json_file):
        fh = open(json_file, 'r')
        dict = json.load(fh)
    else:
        dict = []
    curr_dict = {"epoch":epoch, "loss_trn":loss_trn, "loss_val":loss_val, "acc_metrics":acc_metrics}
    dict.append(curr_dict)
    fh = open(json_file, 'w')
    json.dump(dict, fh, indent=4)
    fh.close()


def train_epoch(model, dataloader, optimizer, device, writer, epoch):
    dataset_size = 0
    running_loss = 0
    criterion = losses.TripletMarginLoss(margin=0.2)
    miner = miners.TripletMarginMiner(margin=0.2,type_of_triplets="hard")

    for i, (data,target,_) in enumerate(dataloader):
        data,target = data.to(device),target.to(device)
        out = model(data)

        hard_triplets = miner(out,target)
        loss = criterion(out, target, hard_triplets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        num_triplets = miner.num_triplets
        running_loss += loss.item() * num_triplets
        dataset_size += num_triplets

        writer.add_scalar("Loss_Train", loss.item(), epoch*len(dataloader)+i)

        if i%100==0:
            print("[Train] Epoch: {}, Batch: {}".format(epoch, i), flush=True)

    epoch_loss = running_loss/dataset_size if dataset_size else 0
    return epoch_loss


def val_epoch(model, dataloader, device, writer, epoch):
    dataset_size = 0
    running_loss = 0
    embedding_db = []
    label_db = []
    criterion = losses.TripletMarginLoss(margin=0.2)
    miner = miners.TripletMarginMiner(margin=0.2,type_of_triplets="hard")

    for i, (data,target,_) in enumerate(dataloader):
        data,target = data.to(device),target.to(device)
        out = model(data)

        hard_triplets = miner(out,target)
        loss = criterion(out, target, hard_triplets)

        num_triplets = miner.num_triplets
        running_loss += loss.item() * num_triplets
        dataset_size += num_triplets

        embedding_db.append(out)
        label_db.append(target)

        if i%100==0:
            print("[Eval] Epoch: {}, Batch: {}".format(epoch, i), flush=True)

    epoch_loss = running_loss/dataset_size if dataset_size else 0
    writer.add_scalar("Loss_Val", epoch_loss, epoch)
    embedding_db = torch.cat(embedding_db)
    label_db = torch.cat(label_db)
    return epoch_loss, embedding_db, label_db


def train_triplet_loss_model(root_dir, checkpoint, batch_size, ckpt_dir, log_dir, num_epochs, learning_rate):
    if checkpoint:
        assert os.path.isfile(checkpoint), "Provided checkpoint not found."
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    dataset = CRC_Feat_Dataset(root_dir=root_dir)

    train_idx,val_idx = train_test_split(range(len(dataset)), test_size=.2, stratify=dataset.targets, random_state=42)
    train_dset = Subset(dataset, train_idx)
    val_dset = Subset(dataset, val_idx)

    train_dl = DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dset, batch_size=batch_size, shuffle=False)

    resnet = models.resnet18(pretrained = True)
    layers = [resnet.layer4[1], resnet.avgpool, nn.Flatten()]
    model = nn.Sequential(*layers)

    if checkpoint:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model).to(device)

    wandb.init(project='ILAWSIA', sync_tensorboard=True)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        model.train()
        loss_trn = train_epoch(model, train_dl, optimizer, device, writer, epoch)
        ckpt_fp = os.path.join(ckpt_dir, "triplet_model_ep{}.pt".format(epoch))
        torch.save(model.module.state_dict(), ckpt_fp)
        model.eval()
        with torch.no_grad():
            loss_val, embedding_db, label_db = val_epoch(model, val_dl, device, writer, epoch)

        acc_cal = AccuracyCalculator()
        acc_metrics = acc_cal.get_accuracy(query=embedding_db.cpu().detach().numpy(), \
            reference=embedding_db.cpu().detach().numpy(), query_labels=label_db.cpu().numpy(), \
            reference_labels=label_db.cpu().numpy(), embeddings_come_from_same_source=True)

        stats_file = os.path.join(log_dir, "stats.json")
        update_stats_json(stats_file, epoch, loss_trn, loss_val, acc_metrics)
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
    parser.add_argument("--log_dir", type=str, default=os.path.join("logs", time))
    args = parser.parse_args()
    print(args,flush=True)
    train_triplet_loss_model(args.root_dir, args.checkpoint, args.batch_size, \
            args.ckpt_dir, args.log_dir, args.num_epochs, args.learning_rate)



if __name__=="__main__":
    main()
