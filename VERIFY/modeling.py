from typing import Any, Optional, List
import os
import vllm
# commented from here
import utils

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
  ) -> None:
    """Initializes a model."""
    self.model_name = model_name
    self.temperature = temperature
    self.max_tokens = max_tokens
    self.show_responses = show_responses
    self.show_prompts = show_prompts
    self.model = self.load_llama(model_name)


  def load_llama(self, model_name: str) -> vllm.LLM: # check the return type and put it here
      # sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)
      model_path = LLAMA_MODELS[model_name]
      print("before load")
      model = vllm.LLM(model=model_path, tensor_parallel_size=4) #, enforce_eager=True)
      print("after load")
      return model

  def generate(
      self,
      prompts: str | list,
      do_debug: bool = False,
      temperature: Optional[float] = None,
      max_tokens: Optional[int] = None,
      timeout: int = 60,
  ) -> str:
    """Generates a response to a prompt."""
    self.model.max_attempts = 1
    self.model.retry_interval = 0
    self.model.timeout = timeout
    # self.model.deployment_name = 'gpt4'
    gen_temp = temperature or self.temperature
    gen_max_tokens = max_tokens or self.max_tokens

    sampling = vllm.SamplingParams(temperature=gen_temp, max_tokens=gen_max_tokens)
    if isinstance(prompts, str):
      prompts = utils.strip_string(prompts)
      prompts = [prompts]
    
    output_list = self.model.generate(prompts, sampling_params=sampling, use_tqdm=False)
    responses = [output.outputs[0].text.strip() for output in output_list]

    if do_debug:
      if self.show_prompts:
        utils.print_color(prompts, 'magenta')
      if self.show_responses:
        utils.print_color(responses, 'cyan')

    return responses if len(prompts) > 1 else responses[0] # batch_size=1
  
  def generate_batched(
      self,
      prompt: List[str],
      do_debug: bool = False,
      temperature: Optional[float] = None,
      max_tokens: Optional[int] = None,
      timeout: int = 60,
  ) -> List[str]:
    """Generates batched response to batched prompts."""
    self.model.max_attempts = 1
    self.model.retry_interval = 0
    self.model.timeout = timeout
    # self.model.deployment_name = 'gpt4'
    prompt = [utils.strip_string(x) for x in prompt]
    gen_temp = temperature or self.temperature
    gen_max_tokens = max_tokens or self.max_tokens
    response, num_attempts = '', 0

    sampling = vllm.SamplingParams(temperature=gen_temp, max_tokens=gen_max_tokens)
    output_list = self.model.generate(prompt, sampling_params=sampling, use_tqdm=False)
    response = [output.outputs[0].text.strip() for output in output_list]

    if do_debug:
      if self.show_prompts:
        utils.print_color(prompt, 'magenta')
      if self.show_responses:
        utils.print_color(response, 'cyan')

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