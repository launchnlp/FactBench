import argparse
import ast
from tqdm import tqdm
import openai
import time
import pandas as pd
import logging

_PROMPT = """Your task is to evaluate how useful and meaningful a user prompts is based on the following 5 criteria:
1. Clarity (0-5): Is the prompt easily understandable without leaving any ambiguity?
2. Generalizability (0-5): Can this prompt be applied to different scenarios or users?
3. Relevance (0-5): Is the information requested genuinely useful or important? Does it have potential interest/value to a broader audience?
4. Actionability (0-5): Is the information requested likely to inform decisions or trigger actions? Does it have practical implications?
5. Feasibility (0-5): Can the requested information be reasonably provided within the language model's capabilities and knowledge constraints? Is it asking for information that exists and is accessible?

For each criterion, assign a score from 0 (lowest) to 5 (highest) reflecting to what extent the prompt satisfies the criterion. \
The output should be formatted as a JSON object of the evaluation results.

Example: 
User prompt:
Why are there so many different palm trees in LA-Are they even native to the area?

Evaluation Results:
{"Clarity": 4, "Generalizability": 2, "Relevance": 3, "Actionability": 2, "Feasibility": 5}

Your Task:
User prompt:
[USER_PROMPT]

Evaluation Results:
"""

# _PROMPT_old='''Your task is to evaluate how useful and meaningful a user prompts is based on the following 7 criteria:

# 1. Clarity (0-5): Is the prompt easily understandable without leaving any ambiguity?

# 2. Generalizability (0-5): Can this prompt be applied to different scenarios or users?

# 3. Specificity (0-5): Does the prompt provide enough detail to generate a focused response?

# 4. Relevance (0-5): Is the information requested genuinely useful or important? Does it have potential interest/value to a broader audience?

# 5. Actionability (0-5): Is the information requested likely to inform decisions or trigger actions? Does it have practical implications?

# 6. Objectivity (0-5): Can it be answered with verifiable data or established knowledge?

# 7. Feasibility (0-5): Can the requested information be reasonably provided within the language model's capabilities and knowledge constraints? Is it asking for information that exists and is accessible?

# For each criterion, assign a score from 0 (lowest) to 5 (highest) and then provide a brief explanation for each score. The output should be formatted as a numbered list with each criterion, its score, and explanation. For example:
# 1. Clarify (4): The prompt is clear and easy to understand. It is not ambiguous.

# User prompt: [USER_PROMPT]'''

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='topic_annotation.out',
    filemode='a'
)

logger = logging.getLogger(__name__)

def parse_list_string(list_string):
    try:
        return ast.literal_eval(list_string)
    except (ValueError, SyntaxError):
        return list_string

openai.api_type = "azure"
openai.api_version = "2024-06-01"
openai.api_key = ""
openai.api_base="https://xxx-openai-service.openai.azure.com/"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default='wildchat', choices=['lmsys', 'wildchat'], help='dataset name')
parser.add_argument('--threshold', type=int, default=0.1, help='threshold for cluster point selection.')
parser.add_argument('--engin', type=str, default='gpt-4-turbo', help='engin')
args = parser.parse_args()

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

# read csv file
input_file_path = "../data/lmsys_data/lmsys_user_prompts_jaccard0.9_w_factually_w_topics_w_model_v2.0.parquet"
df = pd.read_parquet(input_file_path)
factual_df = df[(df["factual_vs_faithful"] == "Factual")]
factual_df = factual_df[factual_df["model"].isin(preferred_models)]
# factual_df = factual_df.groupby('user_prompts').first().reset_index()
user_prompts = factual_df["user_prompts"].unique().tolist()

# for model in preferred_models:
#     print(f'Testing model: {model}')
#     print("num of prompts: ", {len(factual_df[factual_df["model"] == model])})
print("num of prompts: ", len(user_prompts))

out_file_path = input_file_path.replace('.parquet', f'_tier_2_and_3_prompts_evaluated_gpt4.csv')
gpt_deploy_name = args.engin

def calculate_score(prompts):
    prompt_list = []
    for idx, prompt in enumerate(prompts):
        print("PROMPT: ", prompt)
        print("*"*100)
        full_prompt = _PROMPT.replace("[USER_PROMPT]", str(prompt))
        prompt_list.append(full_prompt)
    
    final_results = []
    for prompt in prompt_list:
        while True:
            try:
                response = openai.ChatCompletion.create(
                    engine=gpt_deploy_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=128,
                    temperature=0.0,
                )
                break
            except Exception as e:
                print(e)
                print('error')
                time.sleep(3)
                continue
        res = response['choices'][0]['message']['content'].strip()
        final_results.append(res)
        print(f"RESPONSE: {res}")
        print("*"*100)
    return final_results

BATCH_SIZE = 1
for idx in tqdm(range(0, len(user_prompts), BATCH_SIZE)):
    user_prompts_batch = user_prompts[idx:idx+BATCH_SIZE]
    scores = calculate_score(user_prompts_batch)
    factual_df.loc[factual_df["user_prompts"] == user_prompts[idx], "scores"] = scores[0] # BATCH_SIZE == 1

factual_df.to_csv(out_file_path, index=False)
print('output to', out_file_path)