
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
