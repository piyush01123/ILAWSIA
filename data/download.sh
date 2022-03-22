

# wget https://zenodo.org/record/1214456/files/NCT-CRC-HE-100K.zip
# wget https://zenodo.org/record/1214456/files/CRC-VAL-HE-7K.zip
#
# unzip NCT-CRC-HE-100K.zip
# unzip CRC-VAL-HE-7K.zip

mv  CRC-VAL-HE-7K TestSet
python divide.py --root_dir CRC-HE-7K --query_dir QuerySet --search_dir SearchSet
tree > tree.txt
