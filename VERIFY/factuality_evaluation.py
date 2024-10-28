"""Use a search-augmented LLM to evaluate factuality."""

import collections
import time
import dataclasses
import json
from typing import Any
import argparse
import os
import pandas as pd
from tqdm import tqdm
import logging

# pylint: disable=g-bad-import-order
# commented from here
import modeling
import utils
import decontextualize_fact_
import rate_atomic_fact_
from get_atomic_units import UnitExtraction

IRRELEVANT_LABEL = 'Irrelevant'
SUPPORTED_LABEL = rate_atomic_fact_.SUPPORTED_LABEL
CONTRADICTED_LABEL = rate_atomic_fact_.CONTRADICTED_LABEL
UNVERIFIABLE_LABEL = rate_atomic_fact_.UNVERIFIABLE_LABEL

_MAX_PIPELINE_RETRIES = 3

class CheckedStatement:
  """Class for storing checked statements."""

  def __init__(
      self,
      sentence: str | None,
      label: str | None,
      atomic_fact: str,
      self_contained_atomic_fact: str,
      relevance_data: dict[str, Any] | None = None,
      rate_data: rate_atomic_fact_.FinalAnswer | None = None,
      annotation: str = '',
  ):
    self.sentence = sentence
    self.label = label
    self.atomic_fact = atomic_fact
    self.self_contained_atomic_fact = self_contained_atomic_fact
    self.relevance_data = relevance_data
    self.rate_data = rate_data
    self.annotation = annotation
    self.data = {
        # 'sentence': self.sentence, #uncomment for original implementation
        'label': self.label,
        'atomic_fact': self.atomic_fact,
        'self_contained_atomic_fact': self.self_contained_atomic_fact,
        'relevance_data': self.relevance_data if self.relevance_data else None,
        'rate_data': (
            dataclasses.asdict(self.rate_data) if self.rate_data else None
        ),
        'annotation': self.annotation,
    }


def count_labels(checked_statements: list[CheckedStatement]) -> dict[str, int]:
  """Extract scores from the checked statements for a single response."""
  result_dict = collections.defaultdict(int)

  # Ensure that these labels are in the dictionary
  for label in [SUPPORTED_LABEL, IRRELEVANT_LABEL, CONTRADICTED_LABEL, UNVERIFIABLE_LABEL]:
    result_dict[label] = 0

  for statement in checked_statements:
    if not isinstance(statement, CheckedStatement) or not statement.annotation:
      continue

    if statement.annotation.lower() == SUPPORTED_LABEL.lower():
      result_dict[SUPPORTED_LABEL] += 1
    elif statement.annotation.lower() == IRRELEVANT_LABEL.lower():
      result_dict[IRRELEVANT_LABEL] += 1
    elif statement.annotation.lower() == CONTRADICTED_LABEL.lower():
      result_dict[CONTRADICTED_LABEL] += 1
    elif statement.annotation.lower() == UNVERIFIABLE_LABEL.lower():
      result_dict[UNVERIFIABLE_LABEL] += 1 
    else:
      result_dict[statement.annotation] += 1
      utils.maybe_print_error(
          f'Unknown statement factuality type: {statement.annotation}'
      )

  return dict(result_dict)

     
class FactEval:
  """Class for long-fact evaluation"""
  def __init__(
      self,
      results_dir: str,
      data_path: str,
      cache_dir: str, 
      use_cached_units: bool, 
      model_name = "Llama-3-70B-Instruct",
      batch_size = 256,
      logger = None
  ):
    """
    Initialize the FactEval instance.

    Args:
        data_dir (str): Directory where input data is stored.
        results_dir (str): Directory where results will be stored.
        db_path (str | None): Path to the database file.
        cache_dir (str): Directory where cache files will be stored.
        model_name (str, optional): Name of the model to use. Defaults to "Llama-3-70B-Instruct".
        batch_size (int, optional): Batch size for processing. Defaults to 256.
    """
    self.results_dir = results_dir
    self.db = None
    self.retrieval = None
    self.lm = modeling.Model(model_name)
    self.batch_size = batch_size
    self.logger = logger
    
    self.cache_dir = cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    units_path = os.path.join(self.cache_dir, f"atomic_units_tier_1.json") # change with tiers
    self.unit_extractor = UnitExtraction(data_path, units_path, self.lm, use_cached_units=use_cached_units)
    # print("UNITS extracted!")
  
  
  def get_scores(
      self,
      prompt: str,
      response: str,
      input_index: int) -> None:
    
    # atomic_units = get_atomic_units.atomic_units(response=response, model=self.lm)
    atomic_units = self.unit_extractor.atomic_units(prompt, response, logger=self.logger)
    # print("atomic units: ", atomic_units)
    
    # commented for testing the unit extraction part; uncomment for original implementation
    rating_result = self.classify_relevance_and_rate_super_batched(
        prompt=prompt,
        response=response,
        labels_and_atomic_units=atomic_units["all_factual_units"]
    )
    result_dict = {
        'prompt': prompt, 'response': response, **atomic_units, **rating_result
    }

    output_path = f"{self.results_dir}/result_{input_index}.json"
    if not os.path.exists(self.results_dir):
        os.makedirs(self.results_dir, exist_ok=True)
    with open(output_path, 'w') as f:
      json.dump(result_dict, f, indent=4) 
    
  def classify_relevance_and_rate_super_batched(
      self,
      prompt: str,
      response: str,
      labels_and_atomic_units: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify relevance of and rate all given atomic facts."""
  
    self_contained_unit_dict_batch = (
          decontextualize_fact_.main_batched(
              prompt, response, atomic_unit=[unit_data['atomic_unit'] for unit_data in labels_and_atomic_units], model=self.lm
          )
      )
    # test for unit extraction + self_containement 
    # return {'revised_fact_jsonified_all': self_contained_unit_dict_batch}
    
    unit_data_batch = labels_and_atomic_units
    label_batch = [unit_data['label'] for unit_data in unit_data_batch]

    checked_statement_batch, num_fails = [None for _ in range(len(labels_and_atomic_units))], 0
    past_steps_dict_batch = [{} for _ in range(len(labels_and_atomic_units))]

    atomic_unit_batch = [unit_data['atomic_unit'] for unit_data in unit_data_batch]

    while None in checked_statement_batch and num_fails < _MAX_PIPELINE_RETRIES:
      num_fails += 1
      none_idx = []
      for idx in range(len(labels_and_atomic_units)):
        if checked_statement_batch[idx] is None:
          none_idx.append(idx)
      
      rate_data, past_steps_dict = rate_atomic_fact_.check_atomic_fact_batched(
          self, atomic_fact_lst=[self_contained_unit_dict_batch[tmp_idx]["revised_unit"] for tmp_idx in none_idx], rater=self.lm
      )

      for idx in range(len(none_idx)):
        if not isinstance(rate_data[idx], rate_atomic_fact_.FinalAnswer):
          checked_statement_batch[none_idx[idx]] = None
        else:
          checked_statement_batch[none_idx[idx]] = CheckedStatement(
            sentence=None,
            label=label_batch[none_idx[idx]],
            atomic_fact=atomic_unit_batch[none_idx[idx]],
            self_contained_atomic_fact=self_contained_unit_dict_batch[none_idx[idx]]["revised_unit"],
            # relevance_data=revised_fact_dict_batch[none_idx[idx]],
            rate_data=rate_data[idx],
            annotation=rate_data[idx].answer,
        )
          past_steps_dict_batch[none_idx[idx]] = past_steps_dict[idx]
    
    revised_fact_dict_batch = [self_contained_unit_dict_batch[idx] for idx in range(len(checked_statement_batch)) if isinstance(checked_statement_batch[idx], CheckedStatement)]
    past_steps_dict_batch = [past_steps_dict_batch[idx] for idx in range(len(checked_statement_batch)) if isinstance(checked_statement_batch[idx], CheckedStatement)]
    checked_statement_batch = [checked_statement_batch[idx] for idx in range(len(checked_statement_batch)) if isinstance(checked_statement_batch[idx], CheckedStatement)]

    return {
        'checked_statements': [item.data for item in checked_statement_batch],
        'revised_fact_jsonified_all': revised_fact_dict_batch,
        'past_steps_jsonified_all': past_steps_dict_batch,
        **count_labels(checked_statements=checked_statement_batch),
    }

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  # parser.add_argument('--input_path', type=str,default="data/lmsys_data/final_dataset/tier3_data_collection.csv")
  parser.add_argument('--backbone_llm', type=str, default="Llama-3-70B-Instruct")
  parser.add_argument('--cache_dir', type=str, default="/scratch/wangluxy_root/wangluxy1/farimaf/.cache/long-form-factuality/")
  parser.add_argument('--use_cached_units', type=bool, default=False)
  parser.add_argument('--tier_number', type=int, default=1)
  parser.add_argument('--model_name', type=str, default='gpt4-o', help='model name', choices=['gpt4-o', 'llama3.1_70B_instruct', 'gemini', 'llama3.1_405B_instruct', 'claude-3.5-sonnet', 'commandR+', 'mistral-large-2'])
  parser.add_argument('--num_samples', type=int, default=300) #400 for other tiers
  parser.add_argument('--log_file', type=str, default='logs-s1.out')

  args = parser.parse_args()

  print(f"use_cached_units: {args.use_cached_units}")

  input_path = f"data/lmsys_data/final_dataset/tier_{args.tier_number}/{args.model_name}/generations.csv"
  results_dir = f"data/lmsys_data/benchmarking/BenchCurator/{args.model_name}/tier_{args.tier_number}"
  log_file_path = f"data/lmsys_data/benchmarking/BenchCurator/{args.model_name}/logs/logs-s{args.tier_number}.out"
  for file_path in [results_dir, log_file_path]:
    parent_dir = os.path.dirname(file_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
  logging.basicConfig(
      level=logging.DEBUG,
      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
      filename=log_file_path,
      filemode='w'
  )

  logger = logging.getLogger(__name__)

  fe = FactEval(model_name=args.backbone_llm,
                data_path=input_path,
                results_dir=results_dir,
                cache_dir=args.cache_dir,
                use_cached_units=args.use_cached_units, 
                logger=logger)

  input_df = pd.read_csv(input_path)
  print(f"len dataset: {len(input_df)}")

  for idx in tqdm(input_df.index):
    row = input_df.iloc[idx]
    prompt = row['user_prompts']
    response = str(row[f'{args.model_name}_response'])
    if response:
      fe.get_scores(prompt=prompt, response=response, input_index=idx) 
 