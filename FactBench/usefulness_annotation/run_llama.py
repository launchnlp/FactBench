from vllm import LLM, SamplingParams #!!!
from torch import multiprocessing
from tqdm import tqdm 
import logging
logging.basicConfig(filemode="logs/logs-llama.out", level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import pandas as pd
import os
os.environ['HF_HOME'] = '/scratch/wangluxy_root/wangluxy1/farimaf/models'
# import numpy as np
# from tqdm import tqdm

preferred_models = [
    'gpt-4',
    'claude-2',
    'palm-2',
    'gpt-3.5-turbo',
    'claude-1',
    'claude-instant-1',
    'llama-2-13b-chat',
    'vicuna-33b',
    'guanaco-33b',
    'mpt-30b-chat'
]
llama3_template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Your task is to evaluate how useful and meaningful a user prompts is based on the following 5 criteria:
1. Clarity (0-5): Is the prompt easily understandable without leaving any ambiguity?
2. Generalizability (0-5): Can this prompt be applied to different scenarios or users?
3. Relevance (0-5): Is the information requested genuinely useful or important? Does it have potential interest/value to a broader audience?
4. Actionability (0-5): Is the information requested likely to inform decisions or trigger actions? Does it have practical implications?
5. Feasibility (0-5): Can the requested information be reasonably provided within a language model's capabilities and knowledge constraints? Is it asking for information that exists and is accessible?

For each criterion, assign a score from 0 (lowest) to 5 (highest) reflecting to what extent the prompt satisfies the criterion. \
The output should be formatted as a JSON object of the evaluation scores.

Example 1: 
User prompt:
Why are there so many different palm trees in LA-Are they even native to the area?

Evaluation Results:
{"Clarity": 4, "Generalizability": 2, "Relevance": 3, "Actionability": 2, "Feasibility": 5}

Your Task:
User prompt:
[USER_PROMPT]<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Evaluation Results:
"""

# read csv file
input_file_path = "../data/lmsys_data/lmsys_user_prompts_jaccard0.9_w_factually_w_topics_w_model_v1.1.parquet"
df = pd.read_parquet(input_file_path)
factual_df = df[(df["factual_vs_faithful"] == "Factual")] #& (df["Topics"]!= -1)
factual_df = factual_df[factual_df["model"].isin(preferred_models)]
# factual_df = factual_df.groupby('user_prompts').first().reset_index()
user_prompts = factual_df["user_prompts"].unique().tolist()

# llama3_template_old = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>

# Your task is to evaluate how useful and meaningful a user prompts is based on the following 7 criteria:

# 1. Clarity (0-5): Is the prompt easily understandable without leaving any ambiguity?

# 2. Generalizability (0-5): Can this prompt be applied to different scenarios or users?

# 3. Specificity (0-5): Does the prompt provide enough detail to generate a focused response?

# 4. Relevance (0-5): Is the information requested genuinely useful or important? Does it have potential interest/value to a broader audience?

# 5. Actionability (0-5): Is the information requested likely to inform decisions or trigger actions? Does it have practical implications?

# 6. Objectivity (0-5): Can it be answered with verifiable data or established knowledge?

# 7. Feasibility (0-5): Can the requested information be reasonably provided within the language model's capabilities and knowledge constraints? Is it asking for information that exists and is accessible?

# For each criterion, assign a score from 0 (lowest) to 5 (highest) and then provide a brief explanation for each score. The output should be formatted as a numbered list with each criterion, its score, and explanation. For example:
# 1. Clarify (4): The prompt is clear and easy to understand. It is not ambiguous.

# User prompt: {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''


BATCH_SIZE = 200
MAX_STEPS = 5
multiprocessing.set_start_method('spawn')
model_path="meta-llama/Meta-Llama-3-70B-Instruct"
model = LLM(model=model_path, tensor_parallel_size=4, dtype="float16") 
for idx in tqdm(range(0, len(user_prompts), BATCH_SIZE)):
    # print("prompt: ", user_prompts[idx])
    user_prompts_batch = user_prompts[idx:idx+BATCH_SIZE]
    user_prompts_batch = [llama3_template.replace("[USER_PROMPT]", str(user_prompt)) for user_prompt in user_prompts_batch]
    output_list = model.generate(user_prompts_batch, sampling_params=SamplingParams(max_tokens=50, temperature=0), use_tqdm=False)
    for i in range(len(output_list)):
        current_output = output_list[i].outputs[0].text.strip()
        for j in range(MAX_STEPS):
            if current_output == "":
                logger.info(f"empty output: {current_output}")
                current_output = model.generate([user_prompts_batch[i]], sampling_params=SamplingParams(max_tokens=50, temperature=0), use_tqdm=False)
                current_output = current_output[0].outputs[0].text.strip()
                logger.info(f"filled output? {j}: {current_output}")
            else: 
                break
        logger.info(f"output: {current_output}")
        factual_df.loc[factual_df["user_prompts"] == user_prompts[idx+i], "scores"] = current_output
    # final_ans.extend(output_list)
    # print(output_list[:5], flush=True)
out_file_path = input_file_path.replace('.parquet', f'_prompts_evaluated_llama3_instruct.csv')
factual_df.to_csv(out_file_path, index=False)