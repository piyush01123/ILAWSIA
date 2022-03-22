

"""
Measures peformance against test set
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch
import argparse
import numpy as np
import glob
import os
import time
from sklearn import metrics
import json


NUM_CLASSES = 9


class CRC_Feat_Dataset:
    def __init__(self, root_dir):
        self.root_dir=root_dir
        self.files=sorted(glob.glob(os.path.join(root_dir, '*', '*.npy')))
        self.class2id = {j:i for i, j in enumerate(sorted(os.listdir(root_dir)))}

    def __getitem__(self, index):
        file = self.files[index]
        className = file.split('/')[-2]
        fileName = file.split('/')[-1]
        return torch.Tensor(np.load(file)), self.class2id[className], fileName

    def __len__(self):
        return len(self.files)


def run_test(model, device, dataloader, batch_size, dest_dir):
    model.eval()
    y_gt = []
    y_pred = []
    with torch.no_grad():
        for i, (batch,targets,_) in enumerate(dataloader):
            batch = batch.to(device)
            output = model(batch)
            preds = output.argmax(dim=1, keepdim=True).squeeze()
            y_gt.extend(targets.tolist())
            y_pred.extend(preds.tolist())
            if i%100==0:
                print("[INFO: {}] {}/{} Done.".format(time.strftime("%d-%b-%Y %H:%M:%S"), \
                    i*batch_size+len(batch), len(dataloader.dataset)), flush=True)

    report = metrics.classification_report(y_gt, y_pred, digits=4, output_dict=True, \
                labels=range(NUM_CLASSES), target_names=sorted(os.listdir(dataloader.dataset.root_dir)))
    fh = open(os.path.join(dest_dir, "report.json"), 'w')
    json.dump(report, fh, indent=4)
    fh.close()

    fh = open(os.path.join(dest_dir, 'conf.mat'), 'w')
    fh.write(metrics.confusion_matrix(y_gt, y_pred, labels=range(NUM_CLASSES)).__str__())
    fh.close()


def main():
    parser = argparse.ArgumentParser(description='Measures peformance against test set')
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--export_dir", type=str, default="perf_res")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    print(args,flush=True)

    dataset = CRC_Feat_Dataset(root_dir=args.root_dir)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    resnet = models.resnet18(pretrained = True)
    resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)
    layers = [resnet.layer4[1], resnet.avgpool, nn.Flatten(), resnet.fc]
    model = nn.Sequential(*layers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model).to(device)

    os.makedirs(args.export_dir, exist_ok=True)
    run_test(model, device, dataloader, args.batch_size, args.export_dir)
    print("FIN.", flush=True)


if __name__=="__main__":
    main()
