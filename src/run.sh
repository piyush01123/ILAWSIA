mkdir -p /ssd_scratch/cvit/piyush
rsync -aPz ada:/share3/delta_one/CRC/ /ssd_scratch/cvit/piyush/

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
	--sampler_choice cnfp

