export CUDA_VISIBLE_DEVICES=""

python run_ev_esm.py --zs_model_names="esm,ev" --dataset_list='["DHFR"]' --output_folder="zs-test"
