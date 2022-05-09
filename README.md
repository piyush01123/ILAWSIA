
# ILAWSIA
This is the code for the paper "Interactive Learning for Assisting Whole Slide Image Annotation" published at ACPR 2020.

http://cvit.iiit.ac.in/images/ConferencePapers/2021/Interactive_Learning.pdf

![](https://i.imgur.com/5uWJVQX.png)
The paper has results from 2 datasets: 
- NCT-CRC
- ICIAR

# Get data
```
cd data

wget https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip
wget https://zenodo.org/record/1214456/files/CRC-VAL-HE-7K.zip

unzip NCT-CRC-HE-100K.zip
unzip CRC-VAL-HE-7K.zip

mv  CRC-VAL-HE-7K TestSet

python divide.py --root_dir CRC-HE-7K --query_dir QuerySet --search_dir SearchSet

```
# Create required directories
We need to create result directory, checkpoint directory, log directory and a directory for storing embeddings generated at each iteration.
```
timestamp=`date +%s`
result_dir=data/ILAWSIA_results_$timestamp
ckpt_dir=data/ILAWSIA_ckpt_$timestamp
log_dir=data/ILAWSIA_logs_$timestamp
temp_dbdir=data/ILAWSIA_EmbDB_$timestamp

mkdir -p $result_dir $ckpt_dir $log_dir $temp_dbdir
```

# Choose sampler
Now we need to select the sampler we want to use for the interactive learning
```
sampler_choice=cnfp
```
The choices for sampler and their briefs:
- `entropy` : Sample the images with highest entropy (the images model is most uncertain about)
- `random` : Random sampling
- `front_mid_end` : Sample from the front, mid and end of the ranked nearest neighbor list.
- `cnfp`: Acronym for "Closest Negative Farthest Positive". Sample from the closest negative and the farthest positive samples from the ranked nearest neighbor list. The `negative` and `positive` are the predictions as per classification model trained with already labelled images.
-  `hybrid`: A hybrid of all the above sampling techniques.

# Create the Frozen Query, Search and Test DB
```
  python resnet_featurizer.py \
			--root_dir data/QuerySet \
			--dest_dir data/QueryDBFrozen

  python resnet_featurizer.py \
			--root_dir data/SearchSet \
			--dest_dir data/SearchDBFrozen

  python resnet_featurizer.py \
			--root_dir data/TestSet \
			--dest_dir data/TestDBFrozen
```
# Run interactive learning
```

  python interactive_learning.py \
			--query_dir data/QueryDBFrozen \
			--search_dir data/SearchDBFrozen \
			--test_dir data/TestDBFrozen \
			--temp_dbdir $temp_dbdir/EmbDB_$sampler_choice \
			--num_sessions 1000 \
			--rounds_per_session 5 \
			--expert_labels_per_round 10 \
			--sampler_choice $sampler_choice \
			--ckpt_dir $ckpt_dir/ckpt_$sampler_choice \
			--result_dir $result_dir/results_$sampler_choice \
			--log_dir $log_dir/logs_$sampler_choice
```

