from typing import Any
import os
import json

import pandas as pd
from tqdm import tqdm

UNIT_EXTRACTION_POROMPT="""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Instructions: 
- Exhaustively break down the following text into independent content units. Each content unit can take one of the following forms:
  a. Fact: An objective piece of information that can be proven or verified.
  b. Claim: A statement or assertion that expresses a position or viewpoint on a particular topic.
  c. Instruction: A directive or guidance on how to perform a specific task.
  d. Data Format: Any content presented in a specific format, including code, mathematical notations, equations, variables, technical symbols, tables, or structured data formats.
  e. Meta Statement: Disclaimers, acknowledgments, or any other statements about the nature of the response or the responder.
  f. Question: A query or inquiry about a particular topic.
  g. Other: Any other relevant content that doesn't fit into the above categories.
- Label each content unit with its corresponding unit type using the format: [content unit]: [content unit type]
- Refer to the following examples to understand the task and output formats. 

Example 1:
TEXT: Zhejiang Huafang Pharmaceutical Co., Ltd. is a leading chemical company based in China that specializes in the research, manufacturing, and sales of various pharmaceutical products, including excipients and intermediates. The company was founded in 2018 and is located in Hangzhou, a city with a rich history in eastern China. Zhejiang Huafang Pharmaceutical Co., Ltd. is committed to providing high-quality products to its customers in the healthcare industry. The company's manufacturing facilities are equipped with state-of-the-art technology and infrastructure that ensure the production of high-quality products. Overall, Zhejiang Huafang Pharmaceutical Co., Ltd. is a reputable pharmaceutical company with a long history of success in the healthcare industry. The company's commitment to quality, innovation, and customer service has made it a leader in the field of pharmaceutical research and development.

UNITS:
- Zhejiang Huafang Pharmaceutical Co., Ltd. is a leading chemical company: Fact
- Zhejiang Huafang Pharmaceutical Co., Ltd. is based in China: Fact
- Zhejiang Huafang Pharmaceutical Co., Ltd. specializes in the research of various pharmaceutical products: Fact
- Zhejiang Huafang Pharmaceutical Co., Ltd. specializes in the manufacturing of various pharmaceutical products: Fact
- Zhejiang Huafang Pharmaceutical Co., Ltd. specializes in the sales of various pharmaceutical products: Fact
- excipients are the pharmaceutical products of the Zhejiang Huafang Pharmaceutical Co., Ltd.: Fact
- intermediates are the pharmaceutical products of the Zhejiang Huafang Pharmaceutical Co., Ltd.: Fact
- The company was founded in 2018: Fact
- The company is located in Hangzhou: Fact
- Hangzhou is a city: Fact
- Hangzhou has a rich history in eastern China: Fact
- Zhejiang Huafang Pharmaceutical Co., Ltd. is committed to providing high-quality products to its customers in the healthcare industry: Claim
- The company's manufacturing facilities are equipped with state-of-the-art technology: Fact
- The company's manufacturing facilities are equipped with state-of-the-art infrastructure: Fact
- The company's manufacturing facilities are equipped with state-of-the-art technology and infrastructure that ensure the production of high-quality products: Claim
- Zhejiang Huafang Pharmaceutical Co., Ltd. is a reputable pharmaceutical company: Claim
- Zhejiang Huafang Pharmaceutical Co., Ltd. has a long history of success in the healthcare industry: Claim
- The company is committed to quality: Claim
- The company is committed to innovation: Claim
- The company is committed to customer service: Claim
- The company's commitment to quality, innovation, and customer service has made it a leader in the field of pharmaceutical research: Claim
- The company's commitment to quality, innovation, and customer service has made it a leader in the field of pharmaceutical development: Claim

Example 2:
TEXT: I'm here to help you make an informed decision. Both the RTX 3060 Ti and RTX 3060 are powerful GPUs, and the difference between them lies in their performance. The RTX 3060 Ti has more CUDA cores (4864 vs 3584) but a lower boost clock speed (1665 MHz vs 1777 MHz) compared to the RTX 3060. In terms of memory bandwidth, the RTX 3060 Ti has a slight edge over the RTX 3060 with a bandwidth of 448 GB/s compared to 360 GB/s. However, the difference is relatively small. It's important to consider other factors such as the power consumption, cooling system, and compatibility with your system when making a decision."

UNITS: 
- I'm here to help you make an informed decision: Meta Statement
- The RTX 3060 Ti is a powerful GPU: Claim
- The RTX 3060 is a powerful GPU: Claim
- The difference between them lies in their performance: Claim
- The RTX 3060 Ti has more CUDA cores compared to the RTX 3060: Fact
- The RTX 3060 Ti has 4864 CUDA cores: Fact
- The RTX 3060 has 3584 CUDA cores: Fact
- The RTX 3060 Ti has a lower boost clock speed compared to the RTX 3060: Fact
- The RTX 3060 Ti has a boost clock speed of 1665 MHz: Fact
- The RTX 3060 has a boost clock speed of 1777 MHz: Fact
- The RTX 3060 Ti has a slight edge over the RTX 3060 in terms of memory bandwidth: Fact
- The RTX 3060 Ti has a memory bandwidth of 448 GB/s: Fact
- The RTX 3060 has a memory bandwidth of 360 GB/s: Fact
- The difference is relatively small: Claim
- It's important to consider other factors such as power consumption when making a decision: Instruction
- It's important to consider other factors such as cooling system when making a decision: Instruction
- It's important to consider other factors such as compatibility with your system when making a decision: Instruction

Your Task:
TEXT: {_RESPONSE_PLACEHOLDER}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

UNITS:
"""

_ATOMIC_UNIT = 'atomic_unit'
_LABEL = 'label'


def text_to_units(text, separator = '- ') -> list[str]:
    parsed_units = []
    parsed_labels = []
    current_unit = []
    for line in text.strip().splitlines():
        line = line.strip()
        
        if line.startswith(separator):
            if current_unit:
                # Process the previous unit if it's completed
                full_unit = "\n".join(current_unit).strip()
                if ": " in full_unit:
                    unit, label = full_unit.rsplit(": ", 1)
                    parsed_units.append(unit.strip())
                    parsed_labels.append(label.strip())
                current_unit = []
            # Add the new line to the current unit (without leading '- ')
            current_unit.append(line[2:].strip())
        else:
            # Continue adding lines to the current unit
            current_unit.append(line.strip())
    
    # Process the last unit
    if current_unit:
        full_unit = "\n".join(current_unit).strip()
        if ": " in full_unit:
            unit, label = full_unit.rsplit(": ", 1)
            parsed_units.append(unit.strip())
            parsed_labels.append(label.strip())
    
    return parsed_units, parsed_labels



def convert_atomic_units_to_dicts_(
    labels: list[str], units: list[str]
) -> list[dict[str, Any]]:
  return [
      {_LABEL: label, _ATOMIC_UNIT: identified_atomic_unit}
      for label, identified_atomic_unit in zip(labels, units)
  ]


class UnitExtraction(object):
  def __init__(self, data_path, cache_path, model, use_cached_units, batch_size=1000): 
    # consider lower batch sizes if not working
    self.cache_path = cache_path
    self.data_path = data_path
    self.batch_size = batch_size
    self.use_cached_units = use_cached_units
    self.lm = model
    if use_cached_units:
      print("loading units cache")
      self.load_cache()

  def load_cache(self):
    if os.path.exists(self.cache_path):
      with open(self.cache_path, "r") as f:
          self.cache = json.load(f)
    else:
      self.cache = {}
      self.create_cache()
      self.save_cache()

  def create_cache(self):
    dataset = pd.read_csv(self.data_path)
    # prompts = dataset['user_prompts'].values.tolist()
    responses = dataset['model_responses'].values.tolist()
    responses = [str(response) for response in responses]
    for idx in tqdm(range(0, len(responses), self.batch_size)):
      user_responses_batch = responses[idx:idx+self.batch_size]
      input_list = []
      for response in user_responses_batch:
        # assert isinstance(response, str), 'generation must be a string'
        input_list.append(UNIT_EXTRACTION_POROMPT.format(_RESPONSE_PLACEHOLDER=response))
      output_list = self.lm.generate(input_list, temperature=0)
      for i, output in enumerate(output_list):
        self.cache[responses[idx+i]] = output
      # if idx % 4:
      #   print(f"prompt: {prompts[idx]}, response: {self.cache[prompts[idx]]}")

  def save_cache(self):
    with open(self.cache_path, "w") as f:
        json.dump(self.cache, f)
  
  def get_atomic_units_from_paragraph(self, prompt, response, logger):
    if self.use_cached_units:
      try:
        output = self.cache[response]
      except:
        prompt_to_send = UNIT_EXTRACTION_POROMPT.format(_RESPONSE_PLACEHOLDER=response)
        output = self.lm.generate(prompt_to_send, temperature=0)
    else: 
      prompt_to_send = UNIT_EXTRACTION_POROMPT.format(_RESPONSE_PLACEHOLDER=response)
      output = self.lm.generate(prompt_to_send, temperature=0)
    logger.info(f"OUTPUT: {output}")
    units, labels = text_to_units(output)
    # logger.info(f"units: {units}, labels: {labels}")
    # logger.info(f"MODEL RESPONSE: {output}")

    return units, labels

  def atomic_units(self, prompt, response, logger):
    units, labels = self.get_atomic_units_from_paragraph(prompt, response, logger)
    print(f"units: {units}, labels: {labels}")
    units_as_dict = convert_atomic_units_to_dicts_(labels, units)
    facts_as_dict = [unit for unit in units_as_dict if unit[_LABEL].lower() in ["fact", "claim"]]
    
    return {
        'num_claims': len(units),
        'atomic_units': units,
        'all_atomic_units': units_as_dict,
        'all_factual_units': facts_as_dict
    }