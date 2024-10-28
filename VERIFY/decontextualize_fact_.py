"""Decontextualize an atomic fact."""

from typing import Any, List
# pylint: disable=g-bad-import-order
import modeling
import utils
import config

SYMBOL = 'Foo'
NOT_SYMBOL = 'Not Foo'


# v6
_DECONTEXT_PROMPT = """<|eot_id|><|start_header_id|>user<|end_header_id|>

You task is to decontextualize a UNIT to make it standalone. \
Each UNIT is an independent content unit extracted from the broader context of a RESPONSE.   

Vague References:
- Pronouns (e.g., "he", "she", "they", "it")
- Demonstrative pronouns (e.g., "this", "that", "these", "those")
- Unknown entities (e.g., "the event", "the research", "the invention")
- Incomplete names (e.g., "Jeff..." or "Bezos..." when referring to Jeff Bezos)

Instructions: 
Follow the steps below for unit decontextualization:
1. If the UNIT contains vague references, minimally revise them with respect to the specific subjects they refer to in the RESPONSE.
2. The decontextualized UNIT should be minimally revised by ONLY resolving vague references. No additional information must be added.
3. UNIT extraction might decompose a conjunctive statement into multiple units (e.g. Democracy treats citizens as equals regardless of their race or religion -> (1) Democracy treats citizens as equals regardless of their race, (2) Democracy treats citizens as equals regardless of their religion). Avoid adding what is potentially part of another UNIT.
4. Provide a reasoning of the revisions you made to the UNIT, justifying each decision.
5. After showing your reasoning, provide the revised unit and wrap it in a markdown code block.

Example 1: 
UNIT:
Acorns is a financial technology company

RESPONSE:
Acorns is a financial technology company founded in 2012 by Walter Cruttenden, \
Jeff Cruttenden, and Mark Dru that provides micro-investing services. The \
company is headquartered in Irvine, California.

REVISED UNIT:
This UNIT does not contain any vague references. Thus, the unit does not require any further decontextualization.
```
Acorns is a financial technology company
```

Example 2: 
UNIT:
The victim had previously suffered a broken wrist.

RESPONSE:
The clip shows the victim, with his arm in a cast, being dragged to the floor \
by his neck as his attacker says "I'll drown you" on a school playing field, while forcing water from a bottle into the victim's mouth, \
simulating waterboarding. The video was filmed in a lunch break. The clip shows the victim walking away, without reacting, as the attacker \
and others can be heard continuing to verbally abuse him. The victim, a Syrian refugee, had previously suffered a broken wrist; this had also been \
investigated by the police, who had interviewed three youths but took no further action.

REVISED UNIT:
The UNIT contains a vague reference, "the victim." This is a reference to an unknown entity, \
since it is unclear who the victim is. From the RESPONSE, we can see that the victim is a Syrian refugee. \
Thus, the vague reference "the victim" should be replaced with "the Syrian refugee victim."
```
The Syrian refugee victim had previously suffered a broken wrist.
```

Example 3:
UNIT:
The difference is relatively small.

RESPONSE:
Both the RTX 3060 Ti and RTX 3060 are powerful GPUs, and the difference between them lies in their performance. \
The RTX 3060 Ti has more CUDA cores (4864 vs 3584) but a lower boost clock speed (1665 MHz vs 1777 MHz) compared to the RTX 3060. \
In terms of memory bandwidth, the RTX 3060 Ti has a slight edge over the RTX 3060 with a bandwidth of 448 GB/s compared to 360 GB/s. \
However, the difference is relatively small and may not be noticeable in real-world applications.

REVISED UNIT:
The UNIT contains a vague reference, "The difference." From the RESPONSE, we can see that the difference is in memory bandwidth between the RTX 3060 Ti and RTX 3060. \
Thus, the vague reference "The difference" should be replaced with "The difference in memory bandwidth between the RTX 3060 Ti and RTX 3060." \
The sentence from which the UNIT is extracted includes coordinating conjunctions that potentially decompose the statement into multiple units. Thus, adding more context to the UNIT is not necessary.
```
The difference in memory bandwidth between the RTX 3060 Ti and RTX 3060 is relatively small.
```

YOUR TASK:
UNIT:
{UNIT}

RESPONSE:
{RESPONSE}

REVISED UNIT:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""



def revise_fact_batched(
    response: List[str],
    atomic_unit: List[str],
    model: modeling.Model,
    max_retries: int = config.max_retries,
) -> tuple[List[str], List[str]]:
  """Modify the atomic unit to be self-contained. [decontextualization]"""
  full_prompt_batch = []
  revised_atomic_fact_batch = []
  for idx in range(len(atomic_unit)):
    full_prompt = _DECONTEXT_PROMPT.format(UNIT=atomic_unit[idx], RESPONSE=response[idx])
    full_prompt = utils.strip_string(full_prompt)
    full_prompt_batch.append(full_prompt)
  
  model_response_batch = model.generate_batched(full_prompt_batch)

  for idx in range(len(atomic_unit)):
    model_response, revised_fact, num_tries = model_response_batch[idx], '', 1
    revised_fact = utils.extract_first_code_block(
        model_response, ignore_language=True
    )
    # logger.info(f"original fact: {atomic_fact[idx]}, revised_fact 0: {revised_fact}")
    while not revised_fact and num_tries <= max_retries:
      model_response = model.generate(full_prompt_batch[idx])
      model_response_batch[idx] = model_response
      revised_fact = utils.extract_first_code_block(
          model_response, ignore_language=True
      )
      # logger.info(f"original fact: {atomic_fact[idx]}, revised_fact {num_tries}: {revised_fact}")
      num_tries += 1
    revised_atomic_fact_batch.append(revised_fact or atomic_unit[idx])

  return revised_atomic_fact_batch, model_response_batch


def main_batched(
    prompt: List[str] | str, response: List[str] | str, atomic_unit: List[str], model: modeling.Model # this can be | something else
# ) -> tuple[List[bool], List[str], List[dict[str, Any]]]:
) -> tuple[List[str], List[dict[str, Any]]]:
  """Check if the fact is relevant and modify it to be self-contained."""
  if isinstance(prompt, str):
    prompt = [prompt] * len(atomic_unit)
  if isinstance(response, str):
    response = [response] * len(atomic_unit)
  
  assert len(prompt) == len(response) == len(atomic_unit)

  revised_units, model_responses = revise_fact_batched(
      response=response, atomic_unit=atomic_unit, model=model
  )
  model_responses_batch = [{'atomic_unit': atomic_unit[idx], "revised_unit": revised_units[idx], "model_response": model_responses[idx]} for idx in range(len(atomic_unit))]
  return model_responses_batch