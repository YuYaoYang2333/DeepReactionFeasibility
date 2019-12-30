


# training and save model

scripts=Model
out_dir=Checkpoints
python $scripts/train_model.py -lhd 128 -da 100 -r 10 -e 20 -o $out_dir/attention_model.pkl


