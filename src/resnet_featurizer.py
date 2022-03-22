

"""
ResNet pretrained features at layer 4.1
Will create numpy files of size (512,7,7).

ResNet overview:
Shape: (3,224,224)
1. resnet.conv1: Conv2d
2. resnet.bn1: BatchNorm2d
3. resnet.relu: ReLU
4. resnet.maxpool: MaxPool2d
Shape: (64,56,56)
5. resnet.layer1: Block1
Shape: (64,56,56)
6. resnet.layer2: Block2
Shape: (128,28,28)
7. resnet.layer3: Block3
Shape: (256,14,14)
8. resnet.layer4: Block4
Shape: (512,7,7)
9. resnet.avgpool: AdaptiveAvgPool2d
Shape: (512,1,1)
10. Flatten
Shape: (512)
11. fc
Shape: (1000)

Each Block consists of two BasicBlocks
Each BasicBlock consists of residual layer with conv+bn+relu components. Here is a demo:

Overall ResNet18
```
x = torch.randn(7,3,224,224)
resnet = models.resnet18()
resnet.eval()
A = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
B = nn.Sequential(resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
C = nn.Sequential(resnet.avgpool, nn.Flatten(), resnet.fc)
torch.allclose(C(B(A(x))), resnet(x))
```

Each layer consists of two BasicBlock:
```
y = A(x)
B1A, B1B = resnet.layer1
torch.allclose(B1B(B1A(y)), resnet.layer1(y))
```

ResNet BasicBlock demo:
```
B1Aseq = nn.Sequential(B1A.conv1, B1A.bn1, B1A.relu, B1A.conv2, B1A.bn2)
torch.allclose(B1A(y), nn.ReLU()(B1Aseq(y) + y))
```
"""



import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch
import argparse
import numpy as np
from PIL import Image
import glob
import os
import time


class CRC_Dataset:
    def __init__(self, root_dir, transform):
        self.root_dir=root_dir
        self.files=sorted(glob.glob(os.path.join(root_dir, '*', '*.tif')))
        self.transform=transform

    def __getitem__(self, index):
        file = self.files[index]
        im = Image.open(file).convert('RGB')
        className = file.split('/')[-2]
        fileName = file.split('/')[-1]
        return self.transform(im), className, fileName

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
                fp = os.path.join(dest_dir, className, fileName.replace(".tif", ".npy"))
                np.save(fp, feature)
            if i%100==0:
                print("[INFO: {}] {}/{} Done.".format(time.strftime("%d-%b-%Y %H:%M:%S"), i*batch_size+len(batch), len(dataloader.dataset)), flush=True)


def main():
    parser = argparse.ArgumentParser(description='Process args for Feature Extraction')
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--dest_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    print(args,flush=True)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.7407,0.5331,0.7060], [0.2048,0.2673,0.1872])
        ])

    dataset = CRC_Dataset(root_dir=args.root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    resnet = models.resnet18(pretrained = True)
    layers = list(resnet.children())[:7] + [resnet.layer4[0]]
    featurizer = nn.Sequential(*layers)

    # Remaining model:
    # layers = [resnet.layer4[1], resnet.avgpool, nn.Flatten(), resnet.fc]
    # model = nn.Sequential(*layers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    featurizer = nn.DataParallel(featurizer).to(device)

    print("Extracting features from {} at {}".format(args.root_dir, args.dest_dir), flush=True)
    extract_features(featurizer, device, dataloader, args.batch_size, args.dest_dir)
    print("FIN.", flush=True)


if __name__=="__main__":
    main()
