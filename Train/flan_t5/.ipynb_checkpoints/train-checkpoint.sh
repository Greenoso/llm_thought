#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/msc_ml/reproduce.py 


#optional/ some problems for loading checkpoint created by accelerate
accelerate 
accelerate launch --config_file /root/autodl-tmp/msc_ml/se.yaml  /root/autodl-tmp/msc_ml/reproduce.py




accelerate launch --config_file /root/autodl-tmp/msc_ml/t5/se.yaml  /root/autodl-tmp/msc_ml/t5/t5_cot.py





accelerate launch --config_file /root/autodl-tmp/msc_ml/llama2.yaml  /root/autodl-tmp/msc_ml/llama2.py




accelerate launch --config_file /root/autodl-tmp/msc_ml/llama_2/llama2.yaml  /root/autodl-tmp/msc_ml/llama_2/llama2.py