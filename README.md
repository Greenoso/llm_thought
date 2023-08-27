# llm_thought


# Training on task SVAMP:

``` shell
cd llm_thought
accelerate launch --config_file Train/SVAMP/llama2.yaml  Train/SVAMP/llama_2_trlx.py --checkpoint_dir=checkpoint_dir --model_path=model_path 

```