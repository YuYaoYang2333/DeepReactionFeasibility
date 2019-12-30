



# run the generate_products.py

data_dir=Dataset
output_dir=Dataset
scripts=Utils

python $scripts/generate_products.py \
	 -i_Br $data_dir/Br.smi \
	-i_Alkyne $data_dir/Alkyne.smi \
	-t SMI -o $output_dir/gen_products.smi \
	# 1>gen.log 2>&1  &

