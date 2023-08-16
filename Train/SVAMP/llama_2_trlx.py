import json
import os
import sys
from typing import Dict,List

import torch
from datasets import load_dataset
from transformers import pipeline

from peft import LoraConfig

import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
import re




################################ path and hyperparameters


checkpoint_dir='/root/autodl-tmp/msc_ml/llama_2/ckpts_svamp'
model_path="/root/autodl-tmp/Llama-2-7b-chat-hf"
batch_size=1


############################### few shot cot examples from gsm8k

cot_examples = """### Instruction:
Given a question, generate some helpful and creative thoughts step by step and then answer the question.

### Examples:

Question: 
There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Thought:
There are 15 trees originally.
Then there were 21 trees after some more were planted.
So there must have been 21 - 15 = 6.
#Answer: 6

Question: 
If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Thought:
There are originally 3 cars.
2 more cars arrive.
3 + 2 = 5.
#Answer: 5

Question: 
Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Thought:
Originally, Leah had 32 chocolates.
Her sister had 42.
So in total they had 32 + 42 = 74.
After eating 35, they had 74 - 35 = 39.
#Answer: 39
"""
############################################## dataset   
def build_dataset(
    dataset_name="ChilleD/SVAMP",
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """


    ds_train = load_dataset(dataset_name, split="train")
    original_columns = ds_train.column_names


    def preprocess_function(examples):

        body=examples["Body"]
        question=examples["Question"]
        query = cot_examples +'\n\n### Input:\n'+ 'Question:\n' + body +'\n'+ question + '\n\n' + "### Response:\n"

        return {"query": query }
    

    ds_train = ds_train.map(
        preprocess_function,
        batched=False,
        #remove_columns=original_columns,
    )


    print(ds_train)
    return ds_train


# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(dataset_name="ChilleD/SVAMP")
prompt_all=dataset["query"]
answer_all=dataset["Answer"]

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])



################################# reward





ANS_RE = re.compile(r"#Answer: (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

def _extract_answer(completion: str):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def _is_correct(completion, answer):
    answer=str(int(answer)) 
    extracted_answer = _extract_answer(completion)
    print('\n%%%%%%%%%%',extracted_answer,'true answer:',answer)

    if extracted_answer == answer:
        return 1.0
    else:
        return 0.0
    






def real_reward( prompts: List[str], outputs: List[str], **kwargs) -> List[float]:
    rewards = []
    for prompt, output in zip(prompts, outputs):

        index = prompt_all.index(prompt)
        answer=answer_all[index]
        reward=_is_correct(output,answer)

        rewards.append(reward)

    return rewards


def metric_answer(samples: List[str], prompts: List[str], outputs: List[str]) -> Dict[str, List[float]]:
    match=[]

    for prompt, output in zip(prompts, outputs):

        index = prompt_all.index(prompt)
        answer=answer_all[index]
        is_match=_is_correct(output,answer)

        match.append(is_match)



    return {"Answer Matching": match}





def llama_config():
    return TRLConfig(
        train=TrainConfig(
            seq_length=1024,
            epochs=10,
            total_steps=6000,
            batch_size=batch_size,
            checkpoint_interval=1000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
            save_best=True,
            checkpoint_dir=checkpoint_dir
        ),
        model=ModelConfig(model_path=model_path, num_layers_unfrozen=2),
        tokenizer=TokenizerConfig(tokenizer_path=model_path, truncation_side="left"),
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1.0e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        ),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=1.0e-5)),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=128,
            chunk_size=4,
            ppo_epochs=4,
            init_kl_coef=0.05,
            target=6,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1,
            scale_reward="ignored",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
                max_new_tokens=256,
                top_k=50,
                temperature=1.0,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.1
            ),
        ),
    )


def main(hparams={}):
    # Merge sweep config with default config if given
    config = TRLConfig.update(llama_config().to_dict(), hparams)
    
    
    

    
    ###################################
    config.model.peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=32,
        bias="none",
        task_type="CAUSAL_LM",
        )
    #####################################

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    trlx.train(
        reward_fn=real_reward,
        metric_fn=metric_answer,
        prompts=prompt_all[:-50],
        eval_prompts=prompt_all[-50:],
        config=config,
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
