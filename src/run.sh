#!/bin/bash
#SBATCH -A delta_one
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
##SBATCH --mail-type=END


mkdir -p /ssd_scratch/cvit/piyush
rsync -aPz ada:/share3/delta_one/CRC/ /ssd_scratch/cvit/piyush/

for sampler_choice in entropy random front_mid_end cnfp hybrid
do
  rm -rf /ssd_scratch/cvit/piyush/*Frozen
  python resnet_featurizer.py \
    --root_dir /ssd_scratch/cvit/piyush/QuerySet \
    --dest_dir /ssd_scratch/cvit/piyush/QueryDBFrozen

  python resnet_featurizer.py \
    --root_dir /ssd_scratch/cvit/piyush/SearchSet \
    --dest_dir /ssd_scratch/cvit/piyush/SearchDBFrozen

  python resnet_featurizer.py \
    --root_dir /ssd_scratch/cvit/piyush/TestSet \
    --dest_dir /ssd_scratch/cvit/piyush/TestDBFrozen

  python interactive_learning.py \
    --query_dir /ssd_scratch/cvit/piyush/QueryDBFrozen \
  	--search_dir /ssd_scratch/cvit/piyush/SearchDBFrozen \
  	--test_dir /ssd_scratch/cvit/piyush/TestDBFrozen \
  	--temp_dbdir /ssd_scratch/cvit/piyush/EmbDB_`date +%s` \
  	--num_sessions 1000 \
  	--rounds_per_session 5 \
  	--expert_labels_per_round 10 \
  	--sampler_choice $sampler_choice \
    --ckpt_dir /ssd_scratch/cvit/piyush/ckpt_$sampler_choice \
    --result_dir results_$sampler_choice
done
