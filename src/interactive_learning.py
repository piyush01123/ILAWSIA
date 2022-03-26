
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
from sklearn.neighbors import NearestNeighbors
import random
import numpy as np
import faiss
from collections import Counter
import time
import utils
import shutil


NUM_CLASSES = 9


def get_query_from_QE(root_dir, session_id):
    classNames = sorted(os.listdir(root_dir))
    class_idx = session_id % len(classNames)
    class_for_QE = classNames[class_idx]

    Q_embs = []
    for fp in sorted(glob.glob(os.path.join(root_dir, class_for_QE, '*.npy'))):
        Q_embs.append(np.load(fp))

    Q_embs = np.stack(Q_embs)
    Q_avg = Q_embs.mean(axis=0)

    print("Session={} Query class={} Num files for QE={}".format(session_id, class_for_QE, len(Q_embs)), flush=True)
    return Q_avg, class_for_QE, class_idx


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


def train_models(root_dir, ckpt_clf_dir, ckpt_met_dir, log_clf_dir, log_met_dir, save_every_epoch):
    files = glob.glob(os.path.join(root_dir,'*','*.npy'))
    labels = [file.split('/')[-2] for file in files]
    train_data_stats = Counter(labels)
    train_data_stats = {k: train_data_stats[k] for k in sorted(train_data_stats.keys())}
    print(train_data_stats, flush=True)

    train_classifier.train_classification_model(root_dir=root_dir, \
            ckpt_dir=ckpt_clf_dir, log_dir=log_clf_dir, \
            save_every_epoch=save_every_epoch)
    train_metric_model.train_triplet_loss_model(root_dir=root_dir, \
            ckpt_dir=ckpt_met_dir, \
            log_dir=log_met_dir, \
            save_every_epoch=save_every_epoch)


def test_performances(root_dir, curr_test_db_dir, export_dir, curr_ckpt_clf_dir, curr_ckpt_met_dir, last_epoch):
    classification_metrics = test_classifier.test_classifier_model(root_dir=root_dir, export_dir=export_dir, \
            checkpoint = os.path.join(curr_ckpt_clf_dir,"classifier_ep{}.pt".format(last_epoch)))
    featurizer_512.run_featurizer(root_dir=root_dir, dest_dir=curr_test_db_dir,\
            checkpoint = os.path.join(curr_ckpt_met_dir, "triplet_model_ep{}.pt".format(last_epoch)))
    plot_tsne.plot_tsne(root_dir=curr_test_db_dir, outfile=os.path.join(export_dir,"tsne_plot.png"))
    retrieval_metrics = utils.calculate_retrieval_metrics(root_dir=curr_test_db_dir)
    metrics = {"classification_metrics": classification_metrics, "retrieval_metrics": retrieval_metrics}
    return metrics


def get_samples_to_label_from_expert(curr_search_db_dir, query_emb, search_dir, checkpoint, sampler_choice, query_class_idx, K):
    search_db_files = sorted(glob.glob(os.path.join(curr_search_db_dir,'*','*.npy')))
    N = len(search_db_files)
    search_db = []
    for fp in search_db_files:
        search_db.append(np.load(fp))
    search_db = np.stack(search_db)

    neigh = NearestNeighbors(n_neighbors=N)
    neigh.fit(search_db)
    _, near_indices = neigh.kneighbors([query_emb], N)
    near_indices = near_indices[0]

    # calculate probs matrix of shape (n_search, n_classes). Assume you do not know the labels. So cannot get accuracy/loss
    predictions, entropies = utils.get_clf_output(root_dir=search_dir, checkpoint=checkpoint)

    # sampling strategies: entropy, random, front-mid-end, cnfp, hybrid
    sampler_dict = {"entropy": entropy_sampler, "random": random_sampler, "front_mid_end": \
                    front_mid_end_sampler, "cnfp": cnfp_sampler, "hybrid": hybrid_sampler}
    sampler = sampler_dict[sampler_choice]
    file_indices = sampler(predictions, entropies, near_indices, query_class_idx, N, K)
    search_dir_files = np.array(sorted(glob.glob(os.path.join(search_dir,'*','*.npy'))))
    return search_dir_files[file_indices]


def simulate_expert_annotation(query_dir, search_dir, samples_given_to_expert, query_class):
    # This simulates the situation when the expert gives the label of images given to  him
    # There could also be another setting where the expert just says whether or not it matches the query image label
    # In the 2nd case for non-matching case, we do not know the correct label, so they could be put into whicover class has max prob
    for file in samples_given_to_expert:
        className = file.split('/')[-2]
        shutil.move(file, os.path.join(query_dir, className))


def run_ilawsia(query_dir, search_dir, test_dir, temp_dbdir, num_sessions, rounds_per_session, \
                expert_labels_per_round, sampler_choice, last_epoch, ckpt_dir, result_dir):
    """
    query_dir,search_dir,test_dir contains frozen (512,7,7) embeddings which are calculated only once.
    These are already created before running this script.
    temp_dbdir contains new DBs of (512,) vectors created during each session+round
    """

    # We will use these variables to store the current locations of 512-dim DBs
    curr_query_db_dir = os.path.join(temp_dbdir, "query_db_before_feedback")
    curr_search_db_dir = os.path.join(temp_dbdir, "search_db_before_feedback")
    curr_test_db_dir = os.path.join(temp_dbdir, "test_db_before_feedback")
    curr_ckpt_clf_dir = os.path.join(ckpt_dir,"ckpt_clf_before_feedback")
    curr_ckpt_met_dir = os.path.join(ckpt_dir,"ckpt_met_before_feedback")
    curr_log_clf_dir = os.path.join(ckpt_dir,"ckpt_clf_before_feedback")
    curr_log_met_dir = os.path.join(ckpt_dir,"ckpt_met_before_feedback")
    curr_result_dir = os.path.join(result_dir,"result_before_feedback")

    all_results_json = os.path.join(result_dir,"all_results.json")

    # Train classifier and metric models from initial query set (10 files per class)
    train_models(root_dir=query_dir, ckpt_clf_dir=curr_ckpt_clf_dir, ckpt_met_dir=curr_ckpt_met_dir, \
                 log_clf_dir=curr_log_clf_dir, log_met_dir=curr_log_met_dir, save_every_epoch=False)

    # Measure performaance from initial model
    metrics = test_performances(root_dir=test_dir, curr_test_db_dir=curr_test_db_dir, export_dir=curr_result_dir, \
                 curr_ckpt_clf_dir=curr_ckpt_clf_dir, curr_ckpt_met_dir=curr_ckpt_met_dir,last_epoch=last_epoch)
    utils.update_json({"session_id":-1,"round":-1, "metrics":metrics}, all_results_json)

    # For each session+round, get Query and Search 512-d features from latest checkpoint.
    # Then run sampler, move stuff from search to query, re-train models and measure performances.
    for session_id in range(num_sessions):
        for round in range(rounds_per_session):
            print("Session: {} Round: {}".format(session_id, round), flush=True)

            curr_query_db_dir = os.path.join(temp_dbdir, "query_db_sess_{}_round_{}".format(session_id,round))
            curr_search_db_dir = os.path.join(temp_dbdir, "search_db_sess_{}_round_{}".format(session_id,round))
            curr_test_db_dir = os.path.join(temp_dbdir, "test_db_sess_{}_round_{}".format(session_id,round))

            featurizer_512.run_featurizer(root_dir=query_dir, dest_dir=curr_query_db_dir, \
                    checkpoint=os.path.join(curr_ckpt_met_dir,"triplet_model_ep{}.pt".format(last_epoch)))
            featurizer_512.run_featurizer(root_dir=search_dir,dest_dir=curr_search_db_dir, \
                    checkpoint=os.path.join(curr_ckpt_met_dir, "triplet_model_ep{}.pt".format(last_epoch)))

            query_emb, query_class, query_class_idx = get_query_from_QE(root_dir=curr_query_db_dir, session_id=session_id)
            samples_given_to_expert = get_samples_to_label_from_expert(curr_search_db_dir, query_emb, \
                    search_dir, os.path.join(curr_ckpt_clf_dir,"classifier_ep{}.pt".format(last_epoch)), \
                    sampler_choice, query_class_idx, expert_labels_per_round)
            print("samples_given_to_expert", samples_given_to_expert)
            simulate_expert_annotation(query_dir, search_dir, samples_given_to_expert, query_class)

            curr_ckpt_clf_dir = os.path.join(ckpt_dir,"ckpt_clf_sess_{}_round_{}".format(session_id,round))
            curr_ckpt_met_dir = os.path.join(ckpt_dir,"ckpt_met_sess_{}_round_{}".format(session_id,round))
            curr_result_dir = os.path.join(result_dir,"result_sess_{}_round_{}".format(session_id,round))

            train_models(root_dir=query_dir, ckpt_clf_dir=curr_ckpt_clf_dir, ckpt_met_dir=curr_ckpt_met_dir, \
                         log_clf_dir=curr_log_clf_dir, log_met_dir=curr_log_met_dir, save_every_epoch=False)
            metrics = test_performances(root_dir=test_dir, curr_test_db_dir=curr_test_db_dir, export_dir=curr_result_dir, \
                         curr_ckpt_clf_dir=curr_ckpt_clf_dir, curr_ckpt_met_dir=curr_ckpt_met_dir,last_epoch=last_epoch)
            utils.update_json({"session_id":session_id, "round":round, "metrics":metrics}, all_results_json)


def main():
    parser = argparse.ArgumentParser(description="Run ILAWSIA")
    parser.add_argument("--query_dir", type=str, required=True)
    parser.add_argument("--search_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--temp_dbdir", type=str, required=True)
    parser.add_argument("--num_sessions", type=int, required=True)
    parser.add_argument("--rounds_per_session", type=int, required=True)
    parser.add_argument("--expert_labels_per_round", type=int, required=True)
    parser.add_argument("--sampler_choice", type=str, required=True)
    parser.add_argument("--last_epoch", type=int, default=49)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    args = parser.parse_args()
    print(args,flush=True)
    run_ilawsia(args.query_dir, args.search_dir, args.test_dir, args.temp_dbdir, args.num_sessions, \
            args.rounds_per_session, args.expert_labels_per_round, args.sampler_choice, args.last_epoch, \
            args.ckpt_dir, args.result_dir)


if __name__=="__main__":
    main()


#####
# query_dir = "/ssd_scratch/cvit/piyush/QueryDB"
# search_dir = "/ssd_scratch/cvit/piyush/SearchDB"
# test_dir = "/ssd_scratch/cvit/piyush/TestDB"
# temp_dbdir = "/ssd_scratch/cvit/piyush/EmbDB"
# ckpt_dir = "/ssd_scratch/cvit/piyush/EmbDB"
# result_dir = "results"
# num_sessions = 1000
# rounds_per_session = 5
# expert_labels_per_round = 10
# sampler_choice = "cnfp"
# last_epoch=49
