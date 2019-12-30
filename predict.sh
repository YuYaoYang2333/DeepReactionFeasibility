

# predict and save prediction

scripts=Model
out_dir=Checkpoints
model_dir=Checkpoints
# data to be predicted in Dataset/

python $scripts/predict.py -model $model_dir/attention_model.pkl -o $out_dir/predictions.csv



