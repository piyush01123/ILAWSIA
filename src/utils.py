

import train_classifier
import test_classifier
import train_metric_model
import plot_tsne
import featurizer_512
import argparse
import os
import glob
from sklearn.model_selection import train_test_split
import random
import numpy as np
import faiss
from collections import Counter
import time
import json


NUM_CLASSES = 9


def update_json(dict, json_file):
    curr_list = []
    if os.path.isfile(json_file):
        fh = open(json_file,'r')
        curr_list.extend(json.load(fh))
    fh = open(json_file, 'w')
    curr_list.append(dict)
    json.dump(curr_list, fh, indent=4)
    fh.close()


def calculate_retrieval_metrics(root_dir, num_neighbors=1000, k_values=[5,10,25]):
    files = sorted(glob.glob(os.path.join(root_dir,'*','*.npy')))
    random.seed(42)
    random.shuffle(files)
    labels = [file.split('/')[-2] for file in files]
    search_files, query_files, search_labels, query_labels = train_test_split(files, labels, test_size=.4, random_state=42)
    query_db, search_db = [], []
    for fp in query_files: query_db.append(np.load(fp))
    for fp in search_files: search_db.append(np.load(fp))
    query_db, search_db = np.stack(query_db), np.stack(search_db)

    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(search_db.shape[1])
    search_db_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    search_db_index.add(search_db)

    distances, indices = search_db_index.search(query_db, num_neighbors)
    pred = np.array(search_labels)[indices]
    gt = np.array(query_labels)[:,None].repeat(num_neighbors,axis=1)
    result = (pred==gt).astype(int)

    counter = Counter(query_labels)
    P_at_K = {"macro_avg":{}, "micro_avg":{}, "class_wise":{}}
    for k in k_values:
        result_each = result[:,:k].mean(axis=1)
        P_at_K["micro_avg"][k] = result_each.mean()

        score = {i: 0 for i in set(query_labels)}
        for a,b in zip(query_labels, result_each): score[a]+=b
        class_wise_p_at_k = {i:score[i]/counter[i] for i in set(query_labels)}
        macro_avg_p_at_k = np.mean(list(class_wise_p_at_k.values()))
        P_at_K["macro_avg"][k] = macro_avg_p_at_k
        P_at_K["class_wise"][k] = class_wise_p_at_k

    A = np.cumsum(result,axis=1)
    B = np.arange(1,num_neighbors+1)[:,None].T.repeat(len(query_db),axis=0)
    AP = [c[r.astype(bool)].mean() for c,r in zip(A/B,result)]
    MAP = np.mean(AP)
    metrics = {"P@K": P_at_K, "MAP": MAP}
    return metrics


def get_clf_output(root_dir, checkpoint, batch_size=64, K=5):
    assert os.path.isfile(checkpoint), "checkpoint file missing."

    from torch.utils.data import DataLoader
    from torchvision import models
    import torch.nn.functional as F
    import torch.nn as nn
    import torch

    dataset = test_classifier.CRC_Feat_Dataset(root_dir=root_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    resnet = models.resnet18(pretrained = True)
    resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)
    layers = [resnet.layer4[1], resnet.avgpool, nn.Flatten(), resnet.fc]
    model = nn.Sequential(*layers)

    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model).to(device)

    model.eval()
    predictions = []
    entropies = []
    with torch.no_grad():
        for i, (batch,_,_) in enumerate(dataloader): # Note that we pretend we do not know the target
            batch = batch.to(device)
            output = model(batch)
            prob = F.softmax(output, dim=1)
            H_batch = - (prob * torch.log2(prob)).sum(axis=1)
            preds_batch = output.argmax(dim=1)
            predictions.extend(preds_batch.tolist())
            entropies.extend(H_batch.tolist())
            if i%100==0:
                print("[INFO: {}] {}/{} Done.".format(time.strftime("%d-%b-%Y %H:%M:%S"), \
                    i*batch_size+len(batch), len(dataloader.dataset)), flush=True)
    return np.array(predictions), np.array(entropies)
