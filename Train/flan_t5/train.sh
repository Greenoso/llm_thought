# use the command to train flan-t5-large(need to adjust some parameters in the begining of llm_thought/Train/flan_t5/flan_t5_train.py for different training settings )


accelerate launch --config_file /root/autodl-tmp/msc_ml/llm_thought/Train/flan_t5/flan_t5.yaml  /root/autodl-tmp/msc_ml/llm_thought/Train/flan_t5/flan_t5_train.py


