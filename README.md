# If running in notebook
cd work/workspace

# Preprocessing command
python cell-classification/preprocessing/merge.py


# Split command
python cell-classification/split/annot_split.py 

# training command
python cell-classification/model/train_frcnn.py -o simple -p /opt/dkube/input/annot.txt --hf --vf --rot --num_epochs 1

# Evaluation command
python cell-classification/model/evaluate.py --path /opt/dkube/input/annot.txt

# Finish