
"""
ILAWSIA
"""

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


NUM_CLASSES = 9


def get_query_from_QE(root_dir, session):
    classNames = sorted(os.listdir(root_dir))
    class_idx = session%len(classNames)
    class_for_QE = classNames[class_idx]

    Q_embs = []
    for fp in sorted(glob.glob(os.path.join(root_dir, class_for_QE, '*.npy'))):
        Q_embs.append(np.load(fp))

    Q_embs = np.stack(Q_embs)
    Q_avg = Q_embs.mean(axis=0)

    print("Session={} Query class={} Num files for QE={}".format(session, class_for_QE, len(Q_embs)), flush=True)
    return Q_avg, class_for_QE, class_idx


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


def run_ilawsia(query_dir, search_dir, test_dir, db_dir, num_neighbors):
    train_classifier.train_classification_model(root_dir=query_dir, \
            ckpt_dir="ckpt_clf_session_1_round_1", \
            log_dir="logs_clf_session_1_round_1", \
            save_every_epoch=False)
    train_metric_model.train_triplet_loss_model(root_dir=query_dir, \
            ckpt_dir="ckpt_met_session_1_round_1", \
            log_dir="logs_met_session_1_round_1", \
            save_every_epoch=False)
    test_classifier.test_classifier_model(root_dir=test_dir, \
            export_dir="result_session_1_round_1", \
            checkpoint="ckpt_clf_session_1_round_1/classifier_ep49.pt")
    featurizer_512.run_featurizer(root_dir=test_dir, \
            dest_dir=os.path.join(db_dir, "testdb_session_1_round_1"), \
            checkpoint="ckpt_met_session_1_round_1/triplet_model_ep49.pt")
    plot_tsne.plot_tsne(root_dir=os.path.join(db_dir, "testdb_session_1_round_1"), \
            outfile="result_session_1_round_1/tsne_plot.png")

    featurizer_512.run_featurizer(root_dir=query_dir, \
            dest_dir=os.path.join(db_dir, "querydb_session_1_round_1"), \
            checkpoint="ckpt_met_session_1_round_1/triplet_model_ep49.pt")
    featurizer_512.run_featurizer(root_dir=search_dir, \
            dest_dir=os.path.join(db_dir, "searchdb_session_1_round_1"), \
            checkpoint="ckpt_met_session_1_round_1/triplet_model_ep49.pt")




    retrieval_metrics = calculate_retrieval_metrics(root_dir=os.path.join(db_dir, "testdb_session_1_round_1"))

    query_emb, query_class, query_class_idx = get_query_from_QE(root_dir=os.path.join(db_dir, "querydb_session_1_round_1"), session=0)

    search_files = sorted(glob.glob(os.path.join(db_dir, "searchdb_session_1_round_1",'*','*.npy')))
    search_labels = [file.split('/')[-2] for file in search_files]
    search_db = []
    for fp in search_files:
        search_db.append(np.load(fp))
    search_db = np.stack(search_db)

    neigh = NearestNeighbors(n_neighbors=len(search_db))
    neigh.fit(search_db)
    distances, near_indices = neigh.kneighbors([query_emb],len(search_db))
    distances, near_indices = distances[0], indices[0]

    # calculate probs matrix of shape (n_search, n_classes). Assume you do not know the labels. So cannot get accuracy/loss
    predictions, entropies = get_clf_output(root_dir=search_dir, checkpoint="ckpt_clf_session_1_round_1/classifier_ep49.pt")

    # sampling strategies: entropy, random, front-mid-end, cnfp, hybrid

    sampler_dict = {"entropy": entropy_sampler, \
                    "random": random_sampler, \
                    "front_mid_end": front_mid_end_sampler, \
                    "cnfp": cnfp_sampler, \
                    "hybrid": hybrid_sampler"
                   }
    N = len(search_files)
    samples_given_to_expert = sampler_dict[sampler_choice](predictions, entropies, near_indices, query_class_idx, N, K)



def entropy_sampler(predictions, entropies, near_indices, query_class_idx, N, K):
    return entropies.argsort()[::-1][:K]

def random_sampler(predictions, entropies, near_indices, query_class_idx, N, K):
    return np.random.randint(low=0, high=N, size=K)

def front_mid_end_sampler(predictions, entropies, near_indices, query_class_idx, N, K):
    mid = K//3
    front = (K-K//3)//2
    end = K-(front+mid)
    front_indices = near_indices[:front]
    mid_indices = near_indices[N//2:N//2+mid]
    end_indices = near_indices[-end:]
    return np.concatenate([front_indices,mid_indices,end_indices])

def cnfp_sampler(predictions, entropies, near_indices, query_class_idx, N, K):
    is_positive = predictions==query_class_idx
    near_positives = np.array([x for x in near_indices if is_positive[x]])
    near_negatives = np.array([x for x in near_indices if not is_positive[x]])
    closest_negatives = near_negatives[:K//2] # These are the closest predicted negatives
    farthest_positives = near_positives[::-1][:(K-K//2)] # These are the closest predicted negatives
    return np.concatenate([closest_negatives, farthest_positives])

def hybrid_sampler(predictions, entropies, near_indices, query_class_idx, N, K):
    A = entropy_sampler(predictions, entropies, near_indices, query_class_idx, N, K)
    B = front_mid_end_sampler(predictions, entropies, near_indices, query_class_idx, N, K)
    C = cnfp_sampler(predictions, entropies, near_indices, query_class_idx, N, K)
    return np.random.choice(np.unique(np.concatenate([A,B,C])),K, replace=False)


def main():
    parser = argparse.ArgumentParser(description="Run ILAWSIA")
    parser.add_argument("--query_dir", type=str, required=True)
    parser.add_argument("--search_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--db_dir", type=str, required=True)
    args = parser.parse_args()
    print(args,flush=True)
    run_ilawsia(args.query_dir, args.search_dir, args.test_dir, args.db_dir)


if __name__=="__main__":
    main()


#####
query_dir = "/ssd_scratch/cvit/piyush/QueryDB"
search_dir = "/ssd_scratch/cvit/piyush/SearchDB"
test_dir = "/ssd_scratch/cvit/piyush/TestDB"
db_dir = "/ssd_scratch/cvit/piyush/EmbDB"
