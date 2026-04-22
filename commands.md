==========Enable module==========
Enable GCLS: --use_gcls
Disable GCLS: --no_gcls
Enable Mask Train: --use_mask_train
Enable Mask Infer: --use_mask_infer
==================================================

==========HiCoPro & PrefDial==========
CUDA_VISIBLE_DEVICES=0 nohup python -u HiCoPro.py \
--mode train \
--pretrained models/openmoss/bart-base-chinese \
--train_file data/PrefDial/train_val/path/path_data_train.jsonl \
--val_file data/PrefDial/train_val/path/path_data_val.jsonl \
--graph_file data/PrefDial/train_val/path/graph.json \
--epochs 50 \
--batch_size 64 \
--encoder_lr 7e-5 \
--heads_lr 1e-4 \
--use_gcls \
--use_mask_train \
--use_mask_infer \
--gnn_lr 3e-6 \
--gnn_layers 1 \
--layer_weights "0, 1.0, 1.4, 1.0" \
--max_length 128 \
--save_file_name "HiCoPro" > HiCoPro.log 2>&1 &
==================================================

==========HiCoPro & Amazon_Reviews==========
CUDA_VISIBLE_DEVICES=0 nohup python -u HiCoPro.py \
--mode train \
--pretrained models/facebook/bart-base \
--train_file data/Amazon_Reviews/path/amazon_reviews_train_path.jsonl \
--val_file data/Amazon_Reviews/path/amazon_reviews_val_path.jsonl \
--graph_file data/Amazon_Reviews/path/graph.json \
--epochs 50 \
--batch_size 64 \
--encoder_lr 7e-5 \
--heads_lr 1e-4 \
--use_gcls \
--use_mask_train \
--use_mask_infer \
--gnn_lr 3e-6 \
--gnn_layers 1 \
--layer_weights "0, 1.0, 1.0, 1.0" \
--max_length 256 \
--save_file_name "Amazon_Reviews" > Amazon_Reviews.log 2>&1 &
==================================================

==========HiCoPro & DBPedia==========
CUDA_VISIBLE_DEVICES=0 nohup python -u HiCoPro.py \
--mode train \
--pretrained models/facebook/bart-base \
--train_file data/DBPedia/path/df_train_test_cleaned_path.jsonl \
--val_file data/DBPedia/path/df_validation_cleaned_path.jsonl \
--graph_file data/DBPedia/path/graph.json \
--epochs 50 \
--batch_size 64 \
--encoder_lr 7e-5 \
--heads_lr 1e-4 \
--use_gcls \
--use_mask_train \
--use_mask_infer \
--gnn_lr 3e-6 \
--gnn_layers 1 \
--layer_weights "0, 1.0, 1.0, 1.0" \
--max_length 256 \
--save_file_name "DBPedia" > DBPedia.log 2>&1 &
==================================================
