import os

openai_api_key = ''
serper_api_key = ''
random_seed = 1
model_options = {
    'gpt_4_turbo': 'OPENAI:gpt-4-0125-preview',
    'gpt_4': 'OPENAI:gpt-4-0613',
    'gpt_4_32k': 'OPENAI:gpt-4-32k-0613',
    'gpt_35_turbo': 'OPENAI:gpt-3.5-turbo-0125',
    'gpt_35_turbo_16k': 'OPENAI:gpt-3.5-turbo-16k-0613',
    'claude_3_opus': 'ANTHROPIC:claude-3-opus-20240229',
    'claude_3_sonnet': 'ANTHROPIC:claude-3-sonnet-20240229',
    'claude_3_haiku': 'ANTHROPIC:claude-3-haiku-20240307',
    'claude_21': 'ANTHROPIC:claude-2.1',
    'claude_20': 'ANTHROPIC:claude-2.0',
    'claude_instant': 'ANTHROPIC:claude-instant-1.2',
}

task_options = {}
root_dir = '/'.join(os.path.abspath(__file__).split('/')[:-2])
path_to_data = 'datasets/'
path_to_result = 'results/'


################################################################################
#                               MODEL SETTINGS
# model_short: str = model for fact checking.
# model_temp: float = temperature to use for fact-checking model.
# max_tokens: int = max decode length.
# Currently supported models (copy-paste into `model_short` field): [
#     'gpt_4_turbo',
#     'gpt_4',
#     'gpt_4_32k',
#     'gpt_35_turbo',
#     'gpt_35_turbo_16k',
#     'claude_3_opus',
#     'claude_3_sonnet',
#     'claude_21',
#     'claude_20',
#     'claude_instant',
# ]
################################################################################
model_short = 'gpt_35_turbo'
model_temp = 0.1
max_tokens = 512

################################################################################
#                              SEARCH SETTINGS
# search_type: str = Google Search API used. Choose from ['serper', 'wikipedia'].
# num_searches: int = Number of results to show per search.
################################################################################
search_type = 'serper'
# search_type = 'wikipedia'
num_searches = 3

################################################################################
#                               VERIFY SETTINGS
# max_steps: int = maximum number of break-down steps for factuality check.
# max_retries: int = maximum number of retries when fact checking fails.
################################################################################
max_steps = 5
max_retries = 10

################################################################################
#                         FORCED SETTINGS, DO NOT EDIT
# model: str = overridden by full model name.
################################################################################
model = model_options[model_short]
