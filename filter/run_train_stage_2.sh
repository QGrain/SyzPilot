#!/bin/bash

accelerate launch train_v2.py \
    --total_steps 5000 \
    --batch_size 128  \
    --num_classes 5 \
    --train_stage 2 \
    --base_model_path /opt/syzpilot/models/syzencoder_300w/best_model/ \
    --tokenizer_path /opt/syzpilot/models/customized_tokenizer_224w/ \
    --freeze_layers  \
    --data_dir /artifact/datasets/test_only/test_num_class_5/ \
    --data_idx 1,3,5,7,9,10 \
    --test_data_idx 2,4,6,8 \
    --load_path /opt/syzpilot/filter/logs/TraceClassifier-v2.0-20260124-043138/step-5000.pt
