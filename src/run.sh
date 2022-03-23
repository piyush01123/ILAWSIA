
#!/bin/bash
#SBATCH -A delta_one
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
##SBATCH --mail-type=END

mkdir -p /ssd_scratch/cvit/piyush
rsync -aPz ada:/share3/delta_one/CRC/ /ssd_scratch/cvit/piyush/

python resnet_featurizer.py \
  --root_dir /ssd_scratch/cvit/piyush/QuerySet \
  --dest_dir /ssd_scratch/cvit/piyush/QueryDB

python resnet_featurizer.py \
  --root_dir /ssd_scratch/cvit/piyush/SearchSet \
  --dest_dir /ssd_scratch/cvit/piyush/SearchDB

python resnet_featurizer.py \
  --root_dir /ssd_scratch/cvit/piyush/TestSet \
  --dest_dir /ssd_scratch/cvit/piyush/TestDB


python featurizer_512.py \
  --root_dir /ssd_scratch/cvit/piyush/QueryDB \
  --dest_dir /ssd_scratch/cvit/piyush/Query512DB

python featurizer_512.py \
  --root_dir /ssd_scratch/cvit/piyush/SearchDB \
  --dest_dir /ssd_scratch/cvit/piyush/Search512DB

python featurizer_512.py \
  --root_dir /ssd_scratch/cvit/piyush/TestDB \
  --dest_dir /ssd_scratch/cvit/piyush/Test512DB

mkdir -p results_before_trg results_after_trg

python plot_tsne.py \
  --root_dir /ssd_scratch/cvit/piyush/Test512DB \
  --outfile results_before_trg/tsne_plot.png

python test_classifier.py \
  --root_dir /ssd_scratch/cvit/piyush/TestDB \
  --export_dir results_before_trg

python train_classifier.py \
  --root_dir /ssd_scratch/cvit/piyush/QueryDB \
  --ckpt_dir checkpoints \
  --log_dir logs

python test_classifier.py \
  --root_dir /ssd_scratch/cvit/piyush/TestDB \
  --export_dir results_after_trg \
  --checkpoint checkpoints/classifier_ep49.pt

python train_metric_model.py \
  --root_dir /ssd_scratch/cvit/piyush/QueryDB \
  --ckpt_dir checkpoints_met \
  --log_dir logs_met


python plot_tsne.py \
  --root_dir /ssd_scratch/cvit/piyush/Test512DB \
  --checkpoint checkpoints_met/triplet_model_ep49.pt \
  --outfile results_after_trg/tsne_plot.png
