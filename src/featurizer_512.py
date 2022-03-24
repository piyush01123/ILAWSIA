

"""
(512,7,7) -> (512)
"""

import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch
import argparse
import numpy as np
import glob
import os
import time


class CRC_Feat_Dataset:
    def __init__(self, root_dir):
        self.root_dir=root_dir
        self.files=sorted(glob.glob(os.path.join(root_dir, '*', '*.npy')))

    def __getitem__(self, index):
        file = self.files[index]
        className = file.split('/')[-2]
        fileName = file.split('/')[-1]
        return torch.Tensor(np.load(file)), className, fileName

    def __len__(self):
        return len(self.files)


def extract_features(model, device, dataloader, batch_size, dest_dir):
    model.eval()
    with torch.no_grad():
        for i, (batch,classNames,fileNames) in enumerate(dataloader):
            batch = batch.to(device)
            features = model(batch)
            features = features.detach().cpu().numpy()
            for feature, className, fileName in zip(features, classNames, fileNames):
                os.makedirs(os.path.join(dest_dir, className), exist_ok=True)
                fp = os.path.join(dest_dir, className, fileName.replace(".npy", ".512.npy"))
                np.save(fp, feature)
            if i%100==0:
                print("[INFO: {}] {}/{} Done.".format(time.strftime("%d-%b-%Y %H:%M:%S"), i*batch_size+len(batch), len(dataloader.dataset)), flush=True)


def run_featurizer(root_dir, dest_dir, checkpoint, batch_size=64):
    dataset = CRC_Feat_Dataset(root_dir=root_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    resnet = models.resnet18(pretrained = True)
    layers = [resnet.layer4[1], resnet.avgpool, nn.Flatten()]
    featurizer = nn.Sequential(*layers)

    if checkpoint:
        assert os.path.isfile(checkpoint), "Checkpoint file not found."
        state_dict = torch.load(checkpoint)
        featurizer.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    featurizer = nn.DataParallel(featurizer).to(device)

    print("Extracting features from {} at {}".format(root_dir, dest_dir), flush=True)
    extract_features(featurizer, device, dataloader, batch_size, dest_dir)
    print("FIN.", flush=True)


def main():
    parser = argparse.ArgumentParser(description='Process args for Feature Extraction')
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--dest_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    print(args,flush=True)
    run_featurizer(args.root_dir, args.dest_dir, args.checkpoint, args.batch_size)


if __name__=="__main__":
    main()
