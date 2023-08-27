# if using remote platform autodl
source /etc/network_turbo

accelerate launch --config_file /root/autodl-tmp/msc_ml/llm_thought/Train/SVAMP/llama2.yaml  /root/autodl-tmp/msc_ml/llm_thought/Train/SVAMP/llama_2_trlx.py

accelerate launch --config_file /root/autodl-tmp/msc_ml/cot-rl/config/default_config.yaml /root/autodl-tmp/msc_ml/cot-rl/piqa.py