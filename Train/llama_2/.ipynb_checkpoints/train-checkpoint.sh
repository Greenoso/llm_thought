# if using remote platform autodl
source /etc/network_turbo

accelerate launch --config_file /root/autodl-tmp/msc_ml/llm_thought/Train/llama_2/llama2.yaml  /root/autodl-tmp/msc_ml/llm_thought/Train/llama_2/llama_2_train.py
