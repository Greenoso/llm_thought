# import the libraries
import json
import os
import sys
from typing import Dict, List
import torch
import time
import re
import numpy as np
import random

from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import LoraConfig,  TaskType
import trlx
from trlx.data.configs import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)

from trlx.models.modeling_ppo import PPOConfig


os.environ['WANDB_MODE'] = 'offline'







################################ set path for llm pre-trained-weights and saving directory for checkpoints and hyperparameters

bbh_task='navigate' # task if using bbh dataset, in ['navigate','causal_judgement']
reward_design='qa_accuracy' # in ['qa_accuracy','binary_self_evaluation','confidence_answer','confidence_answer_modified','confidence_qa_se']
bbh_data_dir=f"/root/autodl-tmp/msc_ml/llm_thought/Dataset/BIG-Bench-Hard/bbh/{bbh_task}.json"
svamp_checkpoint_dir='/root/autodl-tmp/msc_ml/llama_2/ckpts_svamp' # path for checkpoints trained with svamp
piqa_checkpoint_dir='/root/autodl-tmp/msc_ml/llama_2/ckpts_piqa' # path for checkpoints trined with piqa
bbh_checkpoint_dir=f'/root/autodl-tmp/msc_ml/t5_large_checkpoints/{bbh_task}/{reward_design}' # path for checkpoints trined with bbh task
model_path="/root/autodl-tmp/flan-t5-large" #path for flan-t5-large pre trined weight
batch_size=16

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
    
set_seed(1002)




default_config = TRLConfig(
    train=TrainConfig(
        seq_length=512,
        epochs=100,
        total_steps=100000,
        # optional gradient accummulation via deepspeed
        batch_size=batch_size,
        checkpoint_interval=1000,
        eval_interval=50,
        pipeline="PromptPipeline",
        trainer="AcceleratePPOTrainer",
        save_best=True,
        tracker="wandb",
        logging_dir='/root/autodl-tmp/msc_ml/llm_thought/Train/flan_t5/wandb/navigate',
        checkpoint_dir=bbh_checkpoint_dir
    ),
    model=ModelConfig(
        model_path="/root/autodl-tmp/flan-t5-large",
        model_arch_type="seq2seq",
    ),
    tokenizer=TokenizerConfig(
        tokenizer_path="/root/autodl-tmp/flan-t5-large",
        truncation_side="right",
    ),
    optimizer=OptimizerConfig(
        name="adamw",
        kwargs={
            "lr": 1.0e-5,
            "betas": [0.9, 0.95],
            "eps": 1.0e-8,
            "weight_decay": 1.0e-6,
        },
    ),
    scheduler=SchedulerConfig(
        name="cosine_annealing",
        kwargs={
            "T_max": 10000,
            "eta_min": 1.0e-6,
        },
    ),
    method=PPOConfig(
        name="PPOConfig",
        ### reduce rollouts due to small dataset
        num_rollouts=128,
        chunk_size=32,
        ppo_epochs=4,
        init_kl_coef=0.1,
        target=6,
        horizon=10000,
        gamma=1,
        lam=0.95,
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=1,
        scale_reward=None,
        ref_mean=None,
        ref_std=None,
        cliprange_reward=10,
        gen_kwargs={
            "max_new_tokens": 256,
        },
        gen_experience_kwargs={
            "max_new_tokens": 256,
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 0.95}
    ),
)


def main(hparams={}):
    print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ task: ',bbh_task,' reward_design: ',reward_design)


    config = TRLConfig.update(default_config, hparams)
    
    config.model.peft_config= LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1
    
    #########################################reward functions
    
    def truncate_after_substring(input_str, substring='answer is'):
        position = input_str.find(substring)
        if position != -1:  # Substring found
            return input_str[:position + len(substring)]
        else:  # Substring not found
            return input_str  # Return the original string if the substring is not found
        
    
    ### reward_se
    def reward_binary_se( prompts: List[str], outputs: List[str], **kwargs) -> List[float]:

        rewards = []
        for q, a in zip(prompts, outputs):
            feedback_prompt = f' The question is: {q}/n/n The answer is: {a}/n/n Is the answer to the question correct? Options: - Yes - No'
            feedback = se_generator(feedback_prompt)[0]['generated_text']  # Assuming 'model' is your trained T5 model
            feedback = feedback.lower().strip()


            if 'yes' in feedback:
                reward = 1.0 
                
            else:
                reward = 0.0

            rewards.append(reward)
            
        return rewards
    
    ### self qa correctness confidence reward
    
    def reward_qa_confidence_se( prompts: List[str], outputs: List[str], **kwargs) -> List[float]:

        rewards = []
        for question, answer in zip(prompts, outputs):
            ### get the binary label for the question-answer pair from answer matching
            generation=''
            # if no legal answer, generation=''
            if 'yes' in answer.lower().strip():
                generation='yes'
            elif 'no' in answer.lower().strip():
                generation='no'


            index = prompt_all_cot.index(question)

            if generation==answer_all[index].lower().strip():
                is_qa_correct=True

            else:
                is_qa_correct=False
            #print('The qa pair is:', is_qa_correct)
                
                
            ### get the logits of relevent token ['Yes','__yes','__Yes'] and ['▁No','▁no','No']
            feedback_prompt = f' The question is: {question}/n/n The answer is: {answer}/n/n Is the answer to the question correct? Options: - Yes - No'
            
            feedback_prompt_ids = tokenizer_se(feedback_prompt, return_tensors="pt").input_ids.cuda()
            gen_tokens = model_se.generate(feedback_prompt_ids, do_sample=False, output_scores =True, return_dict_in_generate = True, temperature=1,max_new_tokens =2 )

            # Decode the generated tokens
            feedback = tokenizer_se.decode(gen_tokens['sequences'][0], skip_special_tokens=True)
            #print('The llm think the qa pair is:',feedback)
            

            # Find the token IDs for 'yes' and 'no'
            yes_token_id = tokenizer_se.convert_tokens_to_ids(['▁Yes','▁yes','Yes'])
            no_token_id = tokenizer_se.convert_tokens_to_ids(['▁No','▁no','No'])


            # Extract the logits for 'yes' and 'no'
            logits_for_yes = gen_tokens['scores'][0][0][yes_token_id] 
            logits_for_no = gen_tokens['scores'][0][0][no_token_id] 
            # Concatenate the logits for 'yes' and 'no'
            all_logits = torch.cat((logits_for_yes, logits_for_no))

            # Apply softmax to convert logits to probabilities
            probabilities = torch.nn.functional.softmax(all_logits, dim=0)

            # Extract probabilities for 'Yes' and 'No'
            probabilities_for_yes = probabilities[:len(logits_for_yes)]
            probabilities_for_no = probabilities[len(logits_for_yes):]

            total_probability_for_yes = probabilities_for_yes.sum().item()
            total_probability_for_no = probabilities_for_no.sum().item()

            #print("Total Probability for 'Yes':", total_probability_for_yes)
            #print("Total Probability for 'No':", total_probability_for_no)
            


            reward = total_probability_for_yes


                

            rewards.append(reward)
            
        return rewards
    
    
    ### qa_accuracy
    def reward_qa_accuracy( prompts: List[str], outputs: List[str], **kwargs) -> List[float]:
        rewards = []
        for prompt, output in zip(prompts, outputs):

            generation=''
            
            # if no legal answer, generation=''
            if 'yes' in output.lower().strip():
                generation='yes'
            elif 'no' in output.lower().strip():
                generation='no'


            index = prompt_all_cot.index(prompt)

            if generation==answer_all[index].lower().strip():
                reward=1.0

            else:
                reward=0.0

            rewards.append(reward)
            
        return rewards
        
    ### confidence for true answer reward
    
    def reward_confidence( prompts: List[str], outputs: List[str], **kwargs) -> List[float]:

        rewards = []
        for question, answer in zip(prompts, outputs):
            ### get the binary label for the question-answer pair from answer matching
            generation=''
            # if no legal answer, generation=''
            if 'yes' in answer.lower().strip():
                generation='yes'
            elif 'no' in answer.lower().strip():
                generation='no'


            index = prompt_all_cot.index(question)

            if generation==answer_all[index].lower().strip():
                is_qa_correct=True

            else:
                is_qa_correct=False
            #print('The qa pair is:', is_qa_correct)
            #truncated_answer_id=tokenizer_se(answer, return_tensors="pt").input_ids.cuda()[0,:-3]
            #truncated_answer=tokenizer_se.decode(truncated_answer_id,skip_special_tokens=True) 
            truncated_answer=truncate_after_substring(answer)

            #print('&&&&&&&&:',truncated_answer)
            ### get the logits of relevent token ['Yes','yes','__yes','__Yes'] and ['▁No','▁no','No']
            feedback_prompt = question+truncated_answer
            
            feedback_prompt_ids = tokenizer_se(feedback_prompt, return_tensors="pt").input_ids.cuda()
            gen_tokens = model_se.generate(feedback_prompt_ids, do_sample=False, output_scores =True, return_dict_in_generate = True, temperature=1,max_new_tokens =2 )

            # Decode the generated tokens
            feedback = tokenizer_se.decode(gen_tokens['sequences'][0], skip_special_tokens=True)

            

            # Find the token IDs for 'yes' and 'no'
            yes_token_id = tokenizer_se.convert_tokens_to_ids(['▁Yes','▁yes','Yes'])
            no_token_id = tokenizer_se.convert_tokens_to_ids(['▁No','▁no','No'])


            # Extract the logits for 'yes' and 'no'
            logits_for_yes = gen_tokens['scores'][0][0][yes_token_id] 
            logits_for_no = gen_tokens['scores'][0][0][no_token_id] 
            

            # Concatenate the logits for 'yes' and 'no'
            all_logits = torch.cat((logits_for_yes, logits_for_no))

            # Apply softmax to convert logits to probabilities
            probabilities = torch.nn.functional.softmax(all_logits, dim=0)

            # Extract probabilities for 'Yes' and 'No'
            probabilities_for_yes = probabilities[:len(logits_for_yes)]
            probabilities_for_no = probabilities[len(logits_for_yes):]

            total_probability_for_yes = probabilities_for_yes.sum().item()
            total_probability_for_no = probabilities_for_no.sum().item()

            print("Total Probability for 'Yes' and 'No':", total_probability_for_yes, total_probability_for_no)

            
            reward=0.0
            
            if is_qa_correct==True:
                if generation=='yes':
                    reward = (1+total_probability_for_yes)/2
                elif generation=='no':
                    reward = (1+total_probability_for_no)/2
                
            else:
                if generation=='yes':
                    reward = (1-total_probability_for_yes)/2
                elif generation=='no':
                    reward = (1-total_probability_for_no)/2

            rewards.append(reward)
            print('$$$$$$$$$',reward)
            
        return rewards    
    
    def reward_confidence_modified( prompts: List[str], outputs: List[str], **kwargs) -> List[float]:


        rewards = []
        for question, answer in zip(prompts, outputs):
            ### get the binary label for the question-answer pair from answer matching
            generation=''
            # if no legal answer, generation=''
            if 'yes' in answer.lower().strip():
                generation='yes'
            elif 'no' in answer.lower().strip():
                generation='no'


            index = prompt_all_cot.index(question)
            
            true_answer=answer_all[index].lower().strip()

            if generation==true_answer:
                is_qa_correct=True

            else:
                is_qa_correct=False
            #print('The qa pair is:', is_qa_correct)
            #truncated_answer_id=tokenizer_se(answer, return_tensors="pt").input_ids.cuda()[0,:-3]
            #truncated_answer=tokenizer_se.decode(truncated_answer_id,skip_special_tokens=True) 
            truncated_answer=truncate_after_substring(answer)

            #print('&&&&&&&&:',truncated_answer)
            ### get the logits of relevent token ['Yes','yes','__yes','__Yes'] and ['▁No','▁no','No']
            feedback_prompt = question+truncated_answer
            
            feedback_prompt_ids = tokenizer_se(feedback_prompt, return_tensors="pt").input_ids.cuda()
            gen_tokens = model_se.generate(feedback_prompt_ids, do_sample=False, output_scores =True, return_dict_in_generate = True, temperature=1,max_new_tokens =2 )

            # Decode the generated tokens
            feedback = tokenizer_se.decode(gen_tokens['sequences'][0], skip_special_tokens=True)

            

            # Find the token IDs for 'yes' and 'no'
            yes_token_id = tokenizer_se.convert_tokens_to_ids(['▁Yes','▁yes','Yes'])
            no_token_id = tokenizer_se.convert_tokens_to_ids(['▁No','▁no','No'])


            # Extract the logits for 'yes' and 'no'
            logits_for_yes = gen_tokens['scores'][0][0][yes_token_id] 
            logits_for_no = gen_tokens['scores'][0][0][no_token_id] 
            

            # Concatenate the logits for 'yes' and 'no'
            all_logits = torch.cat((logits_for_yes, logits_for_no))

            # Apply softmax to convert logits to probabilities
            probabilities = torch.nn.functional.softmax(all_logits, dim=0)

            # Extract probabilities for 'Yes' and 'No'
            probabilities_for_yes = probabilities[:len(logits_for_yes)]
            probabilities_for_no = probabilities[len(logits_for_yes):]

            total_probability_for_yes = probabilities_for_yes.sum().item()
            total_probability_for_no = probabilities_for_no.sum().item()

            #print("Total Probability for 'Yes' and 'No':", total_probability_for_yes, total_probability_for_no)

            
            if true_answer=='yes':
                reward=total_probability_for_yes
            elif true_answer=='no':
                reward=total_probability_for_no
            
            


            rewards.append(reward)
            #print('$$$$$$$$$',reward)
            
        return rewards       

    
    
    ### metric_real_accuracy

    
    def metric_real_accuracy(samples: List[str], prompts: List[str], outputs: List[str]) -> Dict[str, List[float]]:
        match=[]
        
        
        for prompt, output in zip(prompts, outputs):

            generation=''
            
            # if no legal answer, generation=''
            if 'yes' in output.lower().strip():
                generation='yes'
            elif 'no' in output.lower().strip():
                generation='no'


            index = prompt_all_cot.index(prompt)

            if generation==answer_all[index].lower().strip():
                is_match=1.0

            else:
                is_match=0

            match.append(is_match)
        


        return {"Answer Matching": match}
    
    ###########################################
    
    ### reward_design in ['qa_accuracy','binary_self_evaluation','confidence_answer','confidence_qa_se']
    
    if reward_design=='qa_accuracy':
        reward_fn=reward_qa_accuracy
    elif reward_design=='binary_self_evaluation':
        reward_fn=reward_binary_se
    elif reward_design=='confidence_answer':
        reward_fn=reward_confidence
    elif reward_design=='confidence_qa_se':
        reward_fn=reward_qa_confidence_se
    elif reward_design=='confidence_answer_modified':
        reward_fn=reward_confidence_modified
        
        
        
    
    
    
    ############################b
    if reward_design!='qa_accuracy':
        # Load the self evaluation model
        model_se = T5ForConditionalGeneration.from_pretrained("/root/autodl-tmp/flan-t5-large")

        # Load the tokenizer
        tokenizer_se = AutoTokenizer.from_pretrained("/root/autodl-tmp/flan-t5-large")

        # Create the pipeline
        se_generator = pipeline("text2text-generation", model=model_se, tokenizer=tokenizer_se,
                            do_sample= False,
                            max_length=64,
                            eos_token_id= tokenizer_se.eos_token_id,
            device=0 if int(os.environ.get("LOCAL_RANK", 0)) == 0 else -1,)
    #############################e


    
    

    
    #########################b

    ds = load_dataset("json", data_files=f"/root/autodl-tmp/msc_ml/llm_thought/Dataset/BIG-Bench-Hard/bbh/{bbh_task}.json",field="examples")['train']
   
    with open(f'/root/autodl-tmp/msc_ml/llm_thought/Dataset/BIG-Bench-Hard/cot-prompts/{bbh_task}.txt', 'r') as file:
        # Skip the first two lines
        for _ in range(2):
            next(file)

        # Store the rest of the file in a variable
        cot_examples = file.read()

   



    
    answer_all=ds['target']
    
    prompt_all=ds['input']
    # fix the train test split
    train_test_split_id=round(len(answer_all)*0.8)
    print('$$$$$$$$$$$$$$$$$$$$$$$$','training set size:',train_test_split_id)
    
    # ds_split=ds.train_test_split(test_size=0.2)
    #prompt_train=ds_split['train']['input']
    #prompt_test=ds_split['test']['input']
    
    
    prompt_train=prompt_all[:train_test_split_id]
    prompt_test=prompt_all[train_test_split_id:]
    
    prompt_all_cot= ['{} Let’ s think step by step.'.format(prompt.replace('\n', ' ')) for prompt in prompt_all]
    prompt_test_cot= ['{} Let’ s think step by step.'.format(prompt.replace('\n', ' ')) for prompt in prompt_test]
    prompt_train_cot= ['{} Let’ s think step by step.'.format(prompt.replace('\n', ' ')) for prompt in prompt_train]   
    
    

    def replace_newlines(s):
        return re.sub(r'\n+', ' ', s)    
    prompt_all_cot_few_shot= ["Examples: "+cot_examples+"\n\n"+"Question: "+prompt+" Let’ s think step by step.\n\nAnswer:" for prompt in prompt_all]
    prompt_all_cot_few_shot= [replace_newlines(prompt) for prompt in prompt_all_cot]
    prompt_test_cot_few_shot= ["Examples: "+cot_examples+"\n\n"+"Question: "+prompt+" Let’ s think step by step.\n\nAnswer:" for prompt in prompt_test]
    prompt_train_cot_few_shot= ["Examples: "+cot_examples+"\n\n"+"Question: "+prompt+" Let’ s think step by step.\n\nAnswer:" for prompt in prompt_train]    

    ##########################e
    

    trlx.train(
        prompts=prompt_train_cot,
        eval_prompts=prompt_test_cot,
        reward_fn=reward_fn,
        metric_fn=metric_real_accuracy,
        config=config,
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    
    main(hparams)