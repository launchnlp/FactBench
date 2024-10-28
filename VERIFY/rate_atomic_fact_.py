"""Rates a single atomic fact for accuracy."""

import dataclasses
import re
from typing import Any

# pylint: disable=g-bad-import-order
import modeling
import utils
import config
import query_serper
from typing import List
# pylint: enable=g-bad-import-order

SUPPORTED_LABEL = 'Supported'
CONTRADICTED_LABEL = 'Contradicted'
UNVERIFIABLE_LABEL = 'Unverifiable'

_STATEMENT_PLACEHOLDER = '[STATEMENT]'
_KNOWLEDGE_PLACEHOLDER = '[KNOWLEDGE]'
_EARLY_STOP_PLACEHOLDER = '[STOP QUERIES]'

_NEXT_SEARCH_FORMAT_REFINED = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Instructions:
You are engaged in a multi-round process to refine Google Search queries about a given STATEMENT. \
Each round builds upon KNOWLEDGE (a list of previous queries and results, starting empty in round 1). \
Your goal is to improve query quality and relevance over successive rounds.

QUERY CONSTRUCTION CRITERIA: a well-crafted query should:
  - Retrieve information to verify the STATEMENT's factual accuracy.
  - Seek new information not present in the current KNOWLEDGE.
  - Balance specificity for targeted results with breadth to avoid missing critical information.
  - In rounds 2+, leverage insights from earlier queries and outcomes.

Process:
1. Construct a Useful Google Search Query: 
  - Craft a query based on the QUERY CONSTRUCTION CRITERIA.
  - Prioritize natural language queries that a typical user might enter.
  - Use special operators (quotation marks, "site:", Boolean operators, intitle:, etc.) selectively and only when they significantly enhance the query's effectiveness.

2. Provide Query Rationale (2-3 sentences): 
  Explain how this query builds upon previous efforts and/or why it's likely to uncover new, relevant information about the STATEMENT's accuracy.

3. Format Final Query: 
  Present your query in a markdown code block.

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

_FINAL_ANSWER_FORMAT = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Instructions:
You are provided with a STATEMENT and several KNOWLEDGE points. \
Your task is to evaluate the relationship between the STATEMENT and the KNOWLEDGE, following the steps outlined below:

1. Summarize KNOWLEDGE Points: Carefully analyze the KNOWLEDGE points one by one and assess their relevance to the STATEMENT. \
Summarize the main points of the KNOWLEDGE.
2. Evaluate Evidence: Based on your reasoning:
- If the KNOWLEDGE strongly implies or directly supports the STATEMENT, explain the supporting evidence.
- If the KNOWLEDGE contradicts the STATEMENT, identify and explain the conflicting evidence.
- If the KNOWLEDGE is insufficient to confirm or deny the STATEMENT, explain why the evidence is inconclusive.
3. Restate the STATEMENT: After considering the evidence, restate the STATEMENT to maintain clarity.
4. Final Answer: Based on your reasoning and the STATEMENT, determine your final answer. \
Your final answer must be one of the following, wrapped in square brackets:
- [{SUPPORTED_LABEL}] if the STATEMENT is supported by the KNOWLEDGE.
- [{CONTRADICTED_LABEL}] if the STATEMENT is contradicted by the KNOWLEDGE.
- [{UNVERIFIABLE_LABEL}] if the KNOWLEDGE is insufficient to verify the STATEMENT.

KNOWLEDGE: 
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

search_engin = {
  'serper': 'google',
  'wikipedia': 'wiki'
}

@dataclasses.dataclass()
class SearchResult:
  query: str
  result: str
  explanation: str = ''


@dataclasses.dataclass()
class FinalAnswer:
  response: str
  answer: str

# this is the func to change for querying wikipedia
import factuality_evaluation 
def call_search(
    fe_instance: factuality_evaluation.FactEval,
    search_query: str,
    search_type: str = config.search_type,
    num_searches: int = config.num_searches,
    serper_api_key: str = config.serper_api_key,
    search_postamble: str = '',  # ex: 'site:https://en.wikipedia.org'
    topic: str = '', # wikipage title
    ) -> str:
  """Call Google/Wiki Search to get the search result."""
  search_query += f' {search_postamble}' if search_postamble else ''

  if search_type == 'serper':
    serper_searcher = query_serper.SerperAPI(serper_api_key, k=num_searches)
    return serper_searcher.run(search_query, k=num_searches)
  
  elif search_type == 'wikipedia':
    search_results = fe_instance.retrieval.get_passages(topic, search_query, k=num_searches)
    if search_results:
      return ' '.join(search_results) # follwing safe's method
  else:
    raise ValueError(f'Unsupported search type: {search_type}')


def maybe_get_next_search_batched(
    fe_instance: factuality_evaluation.FactEval,
    atomic_fact: List[str],
    past_searches: List[list[SearchResult]],
    model: modeling.Model,
    next_search: List[SearchResult | None],
    ) -> List[SearchResult | None]:
  """Get the next query from the model."""
  none_idx = []
  for idx in range(len(atomic_fact)):
    if next_search[idx] is None:
      none_idx.append(idx)

  full_prompt_batch = []
  for idx in none_idx:
    # knowledge = '\n'.join([s.result for s in past_searches[idx]])
    knowledge = '\n'.join([f"Round {str(i+1)}.\nquery: {s.query}\nresult(s): {s.result}" for i, s in enumerate(past_searches[idx])])
    knowledge = '' if not knowledge else knowledge
    full_prompt = _NEXT_SEARCH_FORMAT_REFINED.replace(_STATEMENT_PLACEHOLDER, atomic_fact[idx])
    full_prompt = full_prompt.replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
    full_prompt = utils.strip_string(full_prompt)
    full_prompt_batch.append(full_prompt)
  
  model_response = model.generate_batched(full_prompt_batch)

  for idx in range(len(none_idx)):
    if _EARLY_STOP_PLACEHOLDER in model_response[idx]:
      next_search[none_idx[idx]] = SearchResult(query="NO QUERY", result="NO RESULT", explanation="Early Stopped")  
    
    else: 
      query = utils.extract_first_code_block(model_response[idx], ignore_language=True)
      if model_response[idx] and query: # if query should suffice!
        next_search[none_idx[idx]] = SearchResult(query=query, result=call_search(fe_instance, query), explanation=model_response[idx])
      else:
        next_search[none_idx[idx]] = None

  return next_search


def maybe_get_final_answer_batched(
    atomic_fact: List[str],
    searches: List[list[SearchResult]],
    model: modeling.Model,
    final_answer: List[FinalAnswer | None],
    ) -> FinalAnswer | None:
  """Get the final answer from the model."""
  none_idx = []
  for idx in range(len(atomic_fact)):
    if final_answer[idx] is None:
      none_idx.append(idx)
  
  full_prompt_batch = []
  for idx in none_idx:
    knowledge = '\n'.join([search.result for search in searches[idx]])
    full_prompt = _FINAL_ANSWER_FORMAT.replace(
        _STATEMENT_PLACEHOLDER, atomic_fact[idx]
    )
    full_prompt = full_prompt.replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
    full_prompt = utils.strip_string(full_prompt)
    full_prompt_batch.append(full_prompt)
  
  model_response = model.generate_batched(full_prompt_batch)

  for idx in range(len(none_idx)):
    answer = utils.extract_first_square_brackets(model_response[idx])
    answer = re.sub(r'[^\w\s]', '', answer).strip()

    if model_response[idx] and answer in [SUPPORTED_LABEL, CONTRADICTED_LABEL, UNVERIFIABLE_LABEL]:
      final_answer[none_idx[idx]] = FinalAnswer(response=model_response[idx], answer=answer)
    else:
      final_answer[none_idx[idx]] = None

  return final_answer


def check_atomic_fact_batched(
    fe_instance: factuality_evaluation.FactEval,
    atomic_fact_lst: List[str],
    rater: modeling.Model,
    max_steps: int = config.max_steps,
    max_retries: int = config.max_retries,
    ) -> tuple[List[FinalAnswer | None], dict[str, Any]]:
  """Check if the given atomic fact is supported."""
  search_results_lst, candid_topics_lst = [[] for _ in range(len(atomic_fact_lst))], [None for _ in range(len(atomic_fact_lst))]
  early_stop_lst = [False for _ in range(len(atomic_fact_lst))]

  search_type = config.search_type
  if search_type == 'wikipedia':
    import wikipedia
    for idx in range(len(candid_topics_lst)):
      query = atomic_fact_lst[idx][:300].lower()
      candid_topics_lst[idx] = wikipedia.search(query)[:max_steps]


  for step_id in range(max_steps):
    # next_search, num_tries = [None for idx in range(len(atomic_fact_lst))], 0
    next_search = [None if not early_stop_lst[idx] else "STOPPED" for idx in range(len(atomic_fact_lst))]
    num_tries = 0
    if search_type == 'serper':
      while None in next_search and num_tries <= max_retries:
        next_search = maybe_get_next_search_batched(fe_instance, atomic_fact_lst, search_results_lst, rater, next_search)
        num_tries += 1
      
      for idx in range(len(search_results_lst)):
        if next_search[idx] is not None and next_search[idx] != "STOPPED":
          if next_search[idx].query == "NO QUERY":
            early_stop_lst[idx] = True
          else: 
            search_results_lst[idx].append(next_search[idx])

    elif search_type == 'wikipedia':
      for idx in range(len(candid_topics_lst)):
        if step_id < len(candid_topics_lst[idx]):
          search_result = call_search(fe_instance, atomic_fact_lst[idx], topic=candid_topics_lst[idx][step_id])
          if search_result is None:
            utils.maybe_print_error(f'{candid_topics_lst[idx][step_id]} not found in wiki dump')
          else:
            search_results_lst[idx].append(SearchResult(query=atomic_fact_lst[idx], result=search_result))

  search_dicts = [{
      f'{search_engin[search_type]}_searches': [dataclasses.asdict(s) for s in search_results]
  } for search_results in search_results_lst]
  # print("search dicts: ", search_dicts)
  final_answer, num_tries = [None for _ in range(len(atomic_fact_lst))], 0

  for idx in range(len(search_results_lst)):
    if not search_results_lst[idx]:
      final_answer[idx] = FinalAnswer(response='No evidence was found', answer=UNVERIFIABLE_LABEL)

  while None in final_answer and num_tries <= max_retries:
    num_tries += 1
    final_answer = maybe_get_final_answer_batched(
        atomic_fact_lst, searches=search_results_lst, model=rater, final_answer=final_answer
    )

  return final_answer, search_dicts

