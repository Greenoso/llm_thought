# if using remote platform autodl
source /etc/network_turbo

accelerate launch --config_file /root/autodl-tmp/msc_ml/llm_thought/Train/question_answer/llama2.yaml  /root/autodl-tmp/msc_ml/llm_thought/Train/question_answer/llama_2_qa_train.py
