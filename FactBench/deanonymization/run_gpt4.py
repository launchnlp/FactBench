import argparse
import ast
from tqdm import tqdm
import openai
import time
import pandas as pd
import logging

def strip_string(s: str) -> str:
  """Strips a string of newlines and spaces."""
  return s.strip(' \n')

_PROMPT_DEANONYMIZE = """Your task is to replace placeholders in the format NAME_x (where x is a number) with appropriate entity names based on the context of the provided text.

Key guidelines:
- Analyze the surrounding text to determine the most likely entity type for each placeholder (person, organization, location, etc.).
- Choose entity names that are contextually relevant and plausible.
- Ensure consistency by using the same entity name for repeated occurrences of the same placeholder.
- Review the final text to ensure coherence and naturalness.

Text: [MODEL_RESPONSE]

OUTPUT:"""
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/deanonymization.out',
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
# user_prompts = factual_df["user_prompts"].unique().tolist()

# for model in preferred_models:
#     print(f'Testing model: {model}')
#     print("num of prompts: ", {len(factual_df[factual_df["model"] == model])})
# print("num of prompts: ", len(user_prompts))

out_file_path = input_file_path.replace('.parquet', f'_deanonymized_final.csv')
gpt_deploy_name = args.engin

def deanonymize_response(model_response):
    model_response = model_response.replace("\\n", "\n").replace('\\"', '\"')
    prompt = _PROMPT_DEANONYMIZE.replace('[MODEL_RESPONSE]', str(model_response))
    prompt = strip_string(prompt)
    while True:
        try:
            response = openai.ChatCompletion.create(
                engine=gpt_deploy_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2048,
                temperature=0.0,
            )
            break
        except Exception as e:
            logger.info(f"ERROR: {e}")
            if "response was filtered due to the prompt triggering Azure OpenAI's content management policy" in str(e):
                return model_response
            else:
                time.sleep(3)
                continue
    message = response['choices'][0]['message']
    logger.info(f"RESPONSE: {message}")
    if "content" in message.keys():
        resp = strip_string(message['content'])
    else: 
        return model_response
        # print(f"MODIFIED RESPONSE: {res}")
        # print("*"*100)
    return resp


for idx in tqdm(range(0, len(factual_df), 1)):
    model_response = factual_df.iloc[idx]['model_responses']
    if "NAME_" in model_response:
        model_response = deanonymize_response(model_response)
        factual_df.loc[factual_df.index[idx], 'model_responses'] = model_response
    if idx % 500 == 0:
        factual_df.to_csv(out_file_path, index=False)

factual_df.to_csv(out_file_path, index=False)
# print('output to', out_file_path)