# import the libraries
import json
import os
import sys
from typing import Dict,List
import random
import numpy as np
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




################################ set path for llm pre-trained-weights and saving directory for checkpoints and hyperparameters


dataset_name='SVAMP' # SVAMP or PIQA
svamp_checkpoint_dir='/root/autodl-tmp/msc_ml/llama_2/ckpts_svamp' # path for checkpoints trained with svamp
piqa_checkpoint_dir='/root/autodl-tmp/msc_ml/llama_2/ckpts_piqa' # path for checkpoints trined with piqa
model_path="/root/autodl-tmp/Llama-2-7b-chat-hf" #path for llama2-7b pre trined weight
batch_size=1

############################### fix random seed

def set_seed(seed: int):
    """
   A function to fix random seed in `numpy`,`random`,  and `torch`.

    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
set_seed(1000)

############################### Instructions prefix to llama for supported datasets

supported_datasets = ['SVAMP', 'PIQA']
if dataset_name not in supported_datasets:
    raise ValueError(f"Unsupported dataset name: {dataset_name}. Supported datasets are {supported_datasets}")


# (optional)cot examples for datasets
svamp_cot_examples = """### Instruction:
Given a question, generate some helpful and creative thoughts step by step and then answer the question.

Question: 
There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Thought:
There are 15 trees originally.
Then there were 21 trees after some more were planted.
So there must have been 21 - 15 = 6.
Answer: 6

Question: 
If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Thought:
There are originally 3 cars.
2 more cars arrive.
3 + 2 = 5.
Answer: 5

Question: 
Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Thought:
Originally, Leah had 32 chocolates.
Her sister had 42.
So in total they had 32 + 42 = 74.
After eating 35, they had 74 - 35 = 39.
Answer: 39
"""

# (optional)cot examples for datasets
piqa_cot_examples = """
Choose the solution to achieve the goal and give answer in bracket:
### goal: When boiling butter, when it's ready, you can: 
### sol1: Pour it onto a plate
### sol2: Pour it into a jar
### Thought: When boiling butter, if it's ready, it is often recommended to use Solution 2: Pour it into a jar. Pouring the boiled butter into a jar allows it to be stored more easily and conveniently, keeping it fresh for later use. Pouring it onto a plate might not be as practical for storage and might lead to quicker spoilage.
### Therefore, the answer is sol [1]

Choose the solution to achieve the goal and give answer in bracket:
### goal: When boiling butter, when it's ready, you can: 
### sol1: Pour it onto a plate
### sol2: Pour it into a jar
### Thought: When boiling butter, if it's ready, it is often recommended to use Solution 2: Pour it into a jar. Pouring the boiled butter into a jar allows it to be stored more easily and conveniently, keeping it fresh for later use. Pouring it onto a plate might not be as practical for storage and might lead to quicker spoilage.
### Therefore, the answer is sol [2]

Choose the solution to achieve the goal and give answer in bracket:
### goal: how do you indent something?: 
### sol1: leave a space before starting the writing
### sol2: press the spacebar
### Thought: Indentation in writing is commonly achieved to visually separate or format content. It is typically done by starting the line with a certain number of spaces or using the "Tab" key on the keyboard. This helps improve readability and organization, especially in paragraphs or programming code.
### Therefore, the answer is sol [1]
"""

# zero shot prompting for svamp
svamp_format="""[INST] «SYS»\n
Given an arithmetic question, generate thoughts about the question step by step and then only give the answer as a number. Please follow the format below:

Thoughts:
<Step by Step Thoughts>
Answer:
<Number>

\n«/SYS»
Qestion:
{}
Thoughts:[/INST]"""
    
    
# zero shot prompting for piqa    
piqa_format="""[INST] «SYS»\n
Given a physical commensense question, generate thoughts about the question step by step, then choose the solution to achieve the goal and give answer in bracket. Please follow the format below:

Thoughts:
<Step by Step Thoughts>
Answer:
The answer is sol <Solution in Bracket>

\n«/SYS»
Qestion: 
{}
Thoughts:[/INST]"""





############################################## dataset  



def build_dataset(
    dataset_name="SVAMP",
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        ds (`dict`):
            The dict for the dataset.
    """
    supported_datasets = ['SVAMP', 'PIQA']
    if dataset_name not in supported_datasets:
        raise ValueError(f"Unsupported dataset name: {dataset_name}. Supported datasets are {supported_datasets}")
    ############# SVAMP
    if dataset_name=='SVAMP':
        dataset_dir="ChilleD/SVAMP"

        ds_all = load_dataset(dataset_dir, split="all")
        original_columns = ds_all.column_names

        def preprocess_function(examples):

            body=examples["Body"]
            question=examples["Question"]
            #query = svamp_cot_examples +'\n\n### Input:\n'+ 'Question:\n' + body +'\n'+ question + '\n\n' + "### Response:\n"
            query = svamp_format.format( body +' '+ question )

            return {"query": query }

        ds_all = ds_all.map(
            preprocess_function,
            batched=False,
            #remove_columns=original_columns,
        )

        print(ds_all)
        return ds_all
    ################# PIQA
    elif dataset_name=='PIQA':
        dataset_dir="piqa"
        
        ds_all = load_dataset(dataset_dir, split="all")
        original_columns = ds_all.column_names

        def preprocess_function(examples):
            string=''
            key = list(examples.keys())
            value = list(examples.values())
            for i, j in zip(key[:-1], value[:-1]):
                string += "\n" + i + ": " + j
            #query = svamp_cot_examples +'\n\n### Input:\n'+ 'Question:\n' + body +'\n'+ question + '\n\n' + "### Response:\n"
            query = piqa_format.format( string )

            return {"query": query }
        

        ds_all = ds_all.map(
            preprocess_function,
            batched=False,
            #remove_columns=original_columns,
        )


        print(ds_all)
        return ds_all   
    


# We retrieve the dict by calling the `build_dataset` function.
dataset = build_dataset(dataset_name=dataset_name)
prompt_all=dataset["query"]
if dataset_name=='SVAMP':
    answer_all=dataset["Answer"]
    checkpoint_dir=svamp_checkpoint_dir
elif dataset_name=='PIQA':
    answer_all=dataset["Answer"]
    checkpoint_dir=piqa_checkpoint_dir
    

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])



################################# reward




if dataset_name=='SVAMP':
    ANS_RE =re.compile(r"Answer:.*?(\$?)(\-?[0-9\.\,]+)", re.DOTALL)
elif dataset_name=='PIQA':
    ANS_RE = re.compile(r"Answer: (\[\d\])")
    
INVALID_ANS = "[invalid]"

    
def _extract_answer(completion: str):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(2).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        print('&&&&&&&&&&&&&&&&&&&&&&&&&',completion)
        return INVALID_ANS

def _is_correct(completion, answer):

    extracted_answer = _extract_answer(completion)
    print('\n%%%%%%%%%%','extracted answer: ',extracted_answer,' true answer:',answer)
    if extracted_answer=="[invalid]":
        return 0.0
    else:  
        try:
            if float(extracted_answer) == float(answer):
                return 1.0
            else:
                return 0.0
        except ValueError:
            # Handle the case where extracted_answer cannot be converted to float
            return 0.0
def svamp_real_reward( prompts: List[str], outputs: List[str], **kwargs) -> List[float]:
    rewards = []
    for prompt, output in zip(prompts, outputs):

        index = prompt_all.index(prompt)
        answer=answer_all[index]
        #print('$$$$$$$$$$$',prompt)
        reward=_is_correct(output,answer)

        rewards.append(reward)

    return rewards


def svamp_metric_answer(samples: List[str], prompts: List[str], outputs: List[str]) -> Dict[str, List[float]]:
    match=[]

    for prompt, output in zip(prompts, outputs):

        index = prompt_all.index(prompt)
        answer=answer_all[index]
        is_match=_is_correct(output,answer)

        match.append(is_match)



    return {"Answer Matching": match}


def piqa_real_reward(outputs, prompts, samples, **kwargs):
    rewards = []
    for output, prompt in zip(outputs, prompts):
        prompt_len = len(prompt)
        gen_output = output
        pattern = r"\[\d\]"
        result = re.findall(pattern=pattern, string=gen_output)
        if len(result) < 1:
            rewards.append(float(0))
            continue
        else:
            result = result[0]
        result = int(result.strip('[').strip(']'))
        pattern = r"goal:.*"
        split_sign = prefix[-300:]
        question = re.findall(pattern=pattern, string=prompt.split(split_sign)[-1])

        if len(question) < 1:

            rewards.append(float(0))
        else:
            idx = prompt_all.index(prompt)
            label = answer_all[idx] + 1
            rewards.append(float(label == result))

            rewards.append(float(1))
        return rewards



def llama_config():
    return TRLConfig(
        train=TrainConfig(
            seq_length=1024,
            epochs=100,
            total_steps=100000,
            batch_size=batch_size,
            # save checkpoint every 10 epoch
            checkpoint_interval=(10*700*4)//(batch_size*8),
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
            save_best=True,
            checkpoint_dir=checkpoint_dir
        ),
        model=ModelConfig(model_path=model_path, 
                          #num_layers_unfrozen=2
                         ),
        tokenizer=TokenizerConfig(tokenizer_path=model_path, truncation_side="left"),
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1.0e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        ),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=1.0e-5)),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=128,
            chunk_size=16,
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
                max_new_tokens=512,
                top_k=50,
                temperature=1.0,
                top_p=0.95,
                do_sample=True
            ),
        ),
    )


def main(hparams={}):

    # Merge sweep config with default config if given
    config = TRLConfig.update(llama_config().to_dict(), hparams)
    
    ###################################
    config.model.peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        )
    #####################################

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1
        
        
    if dataset_name=='SVAMP':
        trlx.train(
            reward_fn=svamp_real_reward,
            metric_fn=svamp_metric_answer,
            prompts=prompt_all[:700],
            eval_prompts=prompt_all[700:800],
            config=config,
        )
    elif dataset_name=='PIQA':
        trlx.train(
            reward_fn=piqa_real_reward,
            prompts=prompt_all[:-50],
            eval_prompts=prompt_all[-50:],
            config=config,
        )
        
if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)