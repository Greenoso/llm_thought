# llm_thought


# Training on task SVAMP:

Change the directories at the begining of Train/question_answer/llama_2_qa_train.py

dataset_name='SVAMP' # SVAMP or PIQA
svamp_checkpoint_dir=<path for checkpoints trained with svamp> # path for checkpoints trained with svamp
model_path=<path for Llama-2-7b-chat-hf> #path for llama2-7b pre trained weight
batch_size=1

then use command:

``` shell
cd llm_thought
accelerate launch --config_file Train/question_answer/llama2.yaml Train/question_answer/llama_2_qa_train.py   

```
