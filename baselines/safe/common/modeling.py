# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Sets up language models to be used."""

from concurrent import futures
import functools
import logging
import os
import threading
import time
import asyncio
import aiohttp
import ssl
import certifi
from typing import Any, Annotated, Optional, List

import anthropic
import openai
import pyglove as pg

# pylint: disable=g-bad-import-order
#from common import modeling_utils
#from common import shared_config
from common import utils
# pylint: enable=g-bad-import-order
#import vllm


_DEBUG_PRINT_LOCK = threading.Lock()
_ANTHROPIC_MODELS = [
    'claude-3-opus-20240229',
    'claude-3-sonnet-20240229',
    'claude-3-haiku-20240307',
    'claude-2.1',
    'claude-2.0',
    'claude-instant-1.2',
]

LLAMA_MODELS = {
  "Llama-3-70B-Instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
  "Llama-3-8B-Instruct": "meta-llama/Meta-Llama-3-8B-Instruct"
}


class Model:
  """Class for storing any single language model."""

  def __init__(
      self,
      model_name: str,
      temperature: float = 0.5,
      max_tokens: int = 2048,
      show_responses: bool = False,
      show_prompts: bool = False,
      whose_key: str = "farima",
  ) -> None:
    """Initializes a model."""
    self.model_name = model_name
    self.temperature = temperature
    self.max_tokens = max_tokens
    self.show_responses = show_responses
    self.show_prompts = show_prompts
    if model_name.lower().startswith('openai:'):
      if whose_key == "farima":
        openai.api_type = "azure"
        openai.api_version = "2024-06-01"
        openai.api_key = ""
        openai.api_base="https://xxx-openai-service.openai.azure.com/"
        self.engine_name = "gpt-35-turbo"

      self.model = self.load(model_name, self.temperature, self.max_tokens)
    # print(self.model)
    elif model_name.lower().startswith('llama'):
      self.model = self.load_llama(model_name)


  # def load_llama(self, model_name: str) -> vllm.LLM: # check the return type and put it here
  #     # sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)
  #     model_path = LLAMA_MODELS[model_name]
  #     model = vllm.LLM(model=model_path, tensor_parallel_size=4, dtype="float16")
  #     return model


  def load(
      self, model_name: str, temperature: float, max_tokens: int
  ):
    """Loads a language model from string representation."""

    if model_name.lower().startswith('openai:'):
      # if not shared_config.openai_api_key:
      #   utils.maybe_print_error('No OpenAI API Key specified.')
      #   utils.stop_all_execution(True)

      # return lf.llms.OpenAI(
      #     model=model_name[7:],
      #     api_key=shared_config.openai_api_key,
      #     sampling_options=sampling,
      # )
      return None
    else:
      raise ValueError(f'ERROR: Unsupported model type: {model_name}.')
  

  # Call ChatGPT with the given prompt, asynchronously.
  async def call_chatgpt_async(self, session, prompt: str, temperature, max_tokens, max_attempts, retry_interval):
    payload = {
        'model': self.model_name[7:],
        'messages': [
            {"role": "user", "content": prompt}
        ]
    }
    result, num_attempts = "", 0 
    while not result and num_attempts < max_attempts:
      try:
        async with session.post(
          url=f'{openai.api_base}openai/deployments/{self.engine_name}/chat/completions?api-version={openai.api_version}',
          headers={"Content-Type": "application/json", "api-key": f"{openai.api_key}", "temperature": str(temperature or self.temperature), "max_tokens": str(max_tokens or self.max_tokens)},
          json=payload,
          ssl=ssl.create_default_context(cafile=certifi.where()),
        ) as response:
          response = await response.json()
          #print(response)
        if "error" in response:
          print(f"OpenAI request failed with error {response['error']}")
          if response['error']['code'] == "content_filter":
            max_attempts = 2
        elif response['choices'][0]["finish_reason"] == 'content_filter':
          max_attempts = 2
        else:
          result = response['choices'][0]['message']['content']
      except Exception as e:
        print(f"Encounter the following error when calling api: {repr(e)}")
        time.sleep(retry_interval)
      num_attempts += 1
    
    return result

  # Call chatGPT for all the given prompts in parallel.
  async def call_chatgpt_bulk(self, prompts, temperature, max_tokens, max_attempts, retry_interval):
    async with aiohttp.ClientSession() as session, asyncio.TaskGroup() as tg:
      responses = [tg.create_task(self.call_chatgpt_async(session, prompt, temperature, max_tokens, max_attempts, retry_interval)) for prompt in prompts]
    return responses

  def generate_batched(
      self,
      prompt_batch: List[str],
      do_debug: bool = False,
      temperature: Optional[float] = None,
      max_tokens: Optional[int] = None,
      max_attempts: int = 100,
      timeout: int = 60,
      retry_interval: int = 10,
  ) -> List[str]:
    """Generates a response to a prompt."""
    # self.model.max_attempts = 1
    # self.model.retry_interval = 0
    # self.model.timeout = timeout
    # self.model.deployment_name = 'gpt4'
    prompt_batch = [prompt for prompt in prompt_batch]
    gen_temp = temperature or self.temperature
    gen_max_tokens = max_tokens or self.max_tokens
    response, num_attempts = '', 0

    if self.model_name.lower().startswith("llama"):
      pass
      # sampling = vllm.SamplingParams(temperature=gen_temp, max_tokens=gen_max_tokens)
      # output_list = self.model.generate([prompt], sampling_params=sampling, use_tqdm=False)
      # response = [output.outputs[0].text.strip() for output in output_list][0] # batch of 1

    else: 
      response = asyncio.run(self.call_chatgpt_bulk(prompt_batch, gen_temp, gen_max_tokens, max_attempts, retry_interval))
      response = [tmp.result() for tmp in response]
    #   while not response and num_attempts < max_attempts:
    #     future = openai.Completion.create(engine=self.model_name[7:],prompt=prompt_batch,max_tokens=gen_max_tokens,temperature=gen_temp)
    #     try:
    #       response = future
    #       tmp_response = [""] * len(prompt_batch)
    #       for choice in response.choices:
    #           tmp_response[choice.index] = choice.text
    #       response = tmp_response
    #       print(response)
    #     except (
    #         openai.error.OpenAIError,
    #         futures.TimeoutError,
    #         anthropic.AnthropicError,
    #     ) as e:
    #       utils.maybe_print_error(e)
    #       time.sleep(retry_interval)

    #     num_attempts += 1

    # if do_debug:
    #   with _DEBUG_PRINT_LOCK:
    #     if self.show_prompts:
    #       utils.print_color(prompt_batch, 'magenta')
    #     if self.show_responses:
    #       utils.print_color(response, 'cyan')

    return response
  

  def print_config(self) -> None:
    settings = {
        'model_name': self.model_name,
        'temperature': self.temperature,
        'max_tokens': self.max_tokens,
        'show_responses': self.show_responses,
        'show_prompts': self.show_prompts,
    }
    print(utils.to_readable_json(settings))

