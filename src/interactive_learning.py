
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

def run_ilawsia(query_dir, search_dir, test_dir, db_dir):
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
