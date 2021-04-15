票据分类-------

训练：
CUDA_VISIBLE_DEVICES='6,7' python train.py --data_dir receipt_data --model_type layoutlm --img_model_type resnet50\
 --model_name_or_path check_points/layout_base --do_lower_case --max_seq_length 512\
 --do_train --num_train_epochs 12 --logging_steps 700 --save_steps 700 --output_dir check_points/recip_cls_v4_smooth_base\
 --labels receipt_data/labels_v2.txt --per_gpu_train_batch_size 4 --overwrite_output_dir --evaluate_during_training
 
测试:
CUDA_VISIBLE_DEVICES='6,7' python test.py --data_dir receipt_data --model_type layoutlm --img_model_type resnet50\
 --model_name_or_path check_points/recip_cls_v4_base/checkpoint-7000 --do_lower_case --max_seq_length 512\
 --do_predict --num_train_epochs 12 --logging_steps 300 --save_steps 300 --output_dir check_points/recip_cls_v4_base\
 --labels receipt_data/labels_v2.txt --per_gpu_eval_batch_size 4 --overwrite_output_dir 
 
 
数据路径：
1. 图片：receipt_data/data_v2
2. OCR识别：receipt_data/json_v2
3. 训练集：receipt_data/train_v2.csv (6292张) ; 验证集：receipt_data/dev_v2.csv (425张)

实验结果:
Test: acc 0.96706, macro_f1 0.9729438346891562, micro_f1 0.9670588235294117

工程化代码：submit/
