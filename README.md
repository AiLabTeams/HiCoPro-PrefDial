完整框架：
HiCoPro — Hierarchical Conversational Profiling Framework

==========HiCoPro(Full)==========
CUDA_VISIBLE_DEVICES=0 nohup python -u HiCoPro.py \
--mode train \
--pretrained models/openmoss/bart-base-chinese \
--train_file data/Ours/train_eval/path/path_data_train.jsonl \
--val_file data/Ours/train_eval/path/path_data_val.jsonl \
--graph_file data/Ours/train_eval/path/graph.json \
--epochs 50 \
--batch_size 64 \
--encoder_lr 6e-5 \
--heads_lr 6e-4 \
--use_node_embeddings \
--gnn_lr 3e-6 \
--use_focal \
--focal_gamma 0.5 \
--focal_alpha 0.3 \
--use_layer_curriculum \
--layer_curriculum_starts "1,1,3,3" \
--layer_curriculum_ramp_epochs 5 \
--layer_weights "0.0, 1.0, 1.4, 1.0" \
--max_length 128 \
--save_file_name "HiCoPro_full_6e-5_6e-4_3e-6" > HiCoPro_Full.log 2>&1 &

==========w/o NE & FL & LCL==========
nohup python -u HiCoPro.py \
--mode train \
--pretrained models/openmoss/bart-base-chinese \
--train_file data/Ours/train_eval/path/path_data_train.jsonl \
--val_file data/Ours/train_eval/path/path_data_val.jsonl \
--graph_file data/Ours/train_eval/path/graph.json \
--epochs 50 \
--batch_size 64 \
--encoder_lr 6e-5 \
--heads_lr 6e-4 \
--layer_weights "0.0, 1.0, 1.4, 1.0" \
--max_length 128 \
--save_file_name "HiCoPro_wo_all_6e-5_6e-4" > HiCoPro_wo_all_6e-5_6e-4.log 2>&1 &

==========Only NE==========
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
--use_node_embeddings \
--gnn_lr 3e-6 \
--gnn_layers 1 \
--layer_weights "0, 1.0, 1.4, 1.0" \
--max_length 256 \
--save_file_name "DB_Only_NE_7e-5_1e-4_3e-6" > DB_Only_NE_7e-5_1e-4_3e-6.log 2>&1 &

==========Only FL==========
CUDA_VISIBLE_DEVICES=0 nohup python -u HiCoPro.py \
--mode train \
--pretrained models/openmoss/bart-base-chinese \
--train_file data/Ours/train_eval/path/path_data_train.jsonl \
--val_file data/Ours/train_eval/path/path_data_val.jsonl \
--graph_file data/Ours/train_eval/path/graph.json \
--epochs 50 \
--batch_size 64 \
--encoder_lr 6e-5 \
--heads_lr 6e-4 \
--use_focal \
--focal_gamma 0.5 \
--focal_alpha 0.25 \
--max_length 128 \
--layer_weights "0, 1.0, 1.4, 1.0" \
--save_file_name "Only_FL_6e-5_6e-4_0.5_0.25" > Only_FL_6e-5_6e-4_0.5_0.25.log 2>&1 &

==========Only LCL==========
CUDA_VISIBLE_DEVICES=0 nohup python -u HiCoPro.py \
--mode train \
--pretrained models/openmoss/bart-base-chinese \
--train_file data/Ours/train_eval/path/path_data_train.jsonl \
--val_file data/Ours/train_eval/path/path_data_val.jsonl \
--graph_file data/Ours/train_eval/path/graph.json \
--epochs 50 \
--batch_size 64 \
--encoder_lr 6e-5 \
--heads_lr 6e-4 \
--use_layer_curriculum \
--layer_curriculum_starts "1,1,3,3" \
--layer_curriculum_ramp_epochs 5 \
--max_length 128 \
--layer_weights "0, 1.0, 1.4, 1.0" \
--save_file_name "Only_LCL_6e-5_6e-4_1133_5" > Only_LCL_6e-5_6e-4_1133_5.log 2>&1 &

==========w/ FL & LCL==========
CUDA_VISIBLE_DEVICES=0 nohup python -u HiCoPro.py \
--mode train \
--pretrained models/openmoss/bart-base-chinese \
--train_file data/Ours/train_eval/path/path_data_train.jsonl \
--val_file data/Ours/train_eval/path/path_data_val.jsonl \
--graph_file data/Ours/train_eval/path/graph.json \
--epochs 50 \
--batch_size 64 \
--encoder_lr 6e-5 \
--heads_lr 6e-4 \
--use_focal \
--focal_gamma 0.5 \
--focal_alpha 0.3 \
--use_layer_curriculum \
--layer_curriculum_starts "1,1,3,3" \
--layer_curriculum_ramp_epochs 5 \
--layer_weights "0, 1.0, 1.4, 1.0" \
--save_file_name "FL_LCL_6e-5_6e-4_0.5_0.3_1133_5" > FL_LCL_6e-5_6e-4_0.5_0.3_1133_5.log 2>&1 &

==========w/ NE & LCL==========
CUDA_VISIBLE_DEVICES=0 nohup python -u HiCoPro.py \
--mode train \
--pretrained models/openmoss/bart-base-chinese \
--train_file data/Ours/train_eval/path/path_data_train.jsonl \
--val_file data/Ours/train_eval/path/path_data_val.jsonl \
--graph_file data/Ours/train_eval/path/graph.json \
--epochs 50 \
--batch_size 64 \
--encoder_lr 7e-5 \
--heads_lr 1e-4 \
--use_node_embeddings \
--gnn_lr 3e-6 \
--use_layer_curriculum \
--layer_curriculum_starts "1,1,3,3" \
--layer_curriculum_ramp_epochs 5 \
--layer_weights "0.0, 1.0, 1.4, 1.0" \
--save_file_name "NE_LCL_7e-5_1e-4_3e-6_1133_5" > NE_LCL_7e-5_1e-4_3e-6_1133_5.log 2>&1 &

==========w/ NE & FL==========
CUDA_VISIBLE_DEVICES=0 nohup python -u HiCoPro.py \
--mode train \
--pretrained models/openmoss/bart-base-chinese \
--train_file data/Ours/train_eval/path/path_data_train.jsonl \
--val_file data/Ours/train_eval/path/path_data_val.jsonl \
--graph_file data/Ours/train_eval/path/graph.json \
--epochs 50 \
--batch_size 64 \
--encoder_lr 7e-5 \
--heads_lr 1e-4 \
--use_node_embeddings \
--gnn_lr 3e-6 \
--use_focal \
--focal_gamma 0.5 \
--focal_alpha 0.3 \
--layer_weights "0.0, 1.0, 1.4, 1.0" \
--save_file_name "NE_FL_7e-5_1e-4_3e-6_0.5_0.3" > NE_FL_7e-5_1e-4_3e-6_0.5_0.3.log 2>&1 &
