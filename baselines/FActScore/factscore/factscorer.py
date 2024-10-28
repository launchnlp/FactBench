import argparse
import string
import json
import numpy as np
import os
import logging
import pandas as pd
from tqdm import tqdm
from factscore.abstain_detection import is_response_abstained
from factscore.atomic_facts import AtomicFactGenerator
#from factscore.clm import CLM
#from factscore.npm import NPM
from factscore.modeling import Model
from factscore.retrieval import DocDB, Retrieval
from filelock import FileLock
import time

class FactScorer(object):

    def __init__(self,
                 model_name="retrieval+ChatGPT",
                 data_dir=".cache/factscore",
                 cache_dir=".cache/factscore",
                 cost_estimate="consider_cache",
                 abstain_detection_type=None,
                 batch_size=256):
        assert model_name in ["retrieval+llama", "retrieval+llama+npm", "retrieval+ChatGPT", "npm", "retrieval+ChatGPT+npm"]
        self.model_name = model_name

        self.db = {}
        self.retrieval = {}
        self.npm = {}
        self.batch_size = batch_size # batch size for retrieval
        self.abstain_detection_type = abstain_detection_type

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.af_generator = None
        self.cost_estimate = cost_estimate

        # if "llama" in model_name:
        #     self.lm = CLM("inst-llama-7B",
        #                   model_dir=os.path.join(model_dir, "inst-llama-7B"),
        #                   cache_file=os.path.join(cache_dir, "inst-llama-7B.pkl"))
        # elif "ChatGPT" in model_name:
        #     self.lm = OpenAIModel("ChatGPT",
        #                           cache_file=os.path.join(cache_dir, "ChatGPT.pkl"),
        #                           key_path=openai_key)
        # else:
        self.lm = Model("openai:gpt-35-turbo")

    def save_cache(self):
        for k, v in self.retrieval.items():
            v.save_cache()

    def register_knowledge_source(self, name="enwiki-20230401", db_path=None, data_path=None):
        assert name not in self.retrieval, f"{name} already registered"
        if db_path is None:
            db_path = os.path.join(self.data_dir, f"{name}.db")

        if data_path is None:
            data_path = os.path.join(self.data_dir, f"{name}.jsonl")

        cache_path = os.path.join(self.cache_dir, f"retrieval-{name}-{str(int(time.time()))}.json")
        embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}-{str(int(time.time()))}.pkl")

        self.db[name] = DocDB(db_path=db_path, data_path=data_path)
        self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, batch_size=self.batch_size)
        if "npm" in self.model_name:
            cache_path = os.path.join(self.cache_dir, f"bm25-{name}.json")
            embed_cache_path = os.path.join(self.cache_dir, f"bm25-{name}.pkl")
            self.npm[name] = NPM(Retrieval(self.db[name], cache_path, embed_cache_path, "bm25"),
                                 "npm-single",
                                 cache_file=os.path.join(self.cache_dir, f"npm-{name}.pkl"))


    def print_cost_estimates(self, total_words, task, model):
        # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        # Number of tokens are roughly 4/3 of the number of words
        total_tokens = total_words * 4.0 / 3

        # https://openai.com/pricing
        # if we use davinci-003, the cost is $0.02 per 1000 tokens
        # if we use gpt-3.5-turbo, the cost is $0.002 per 1000 tokens
        if model == "davinci-003":
            rate = 0.02
        elif model == "gpt-3.5-turbo":
            rate = 0.002

        total_cost = total_tokens * rate / 1000

        # print the total words, tokens, and cost along with rate
        logging.critical("Estimated OpenAI API cost for %s ($%.3f per 1000 tokens): $%.2f for %d words and %d tokens" % (task, rate, total_cost, total_words, total_tokens))

    def get_score(self,
                  topics,
                  generations,
                  gamma=10,
                  atomic_facts=None,
                  knowledge_source=None,
                  verbose=False,
                  input_indexs=None,
                  start_idx=None,
                  end_idx=None):
        if knowledge_source is None:
            # use the default knowledge source
            knowledge_source = "enwiki-20230401"
        
        if input_indexs is None:
            input_indexs = [tmp for tmp in range(len(topics))]

        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)

        if type(topics)==type(generations)==str:
            topics = [topics]
            generations = [generations]
        else:
            assert type(topics)==type(generations)==list, "`topics` and `generations` should be lists."
            assert len(topics)==len(generations), "`topics` and `generations` should have the same length"

        if atomic_facts is not None:
            assert len(topics)==len(atomic_facts), "`topics` and `atomic_facts` should have the same length"
        else:
            if self.af_generator is None:
                self.af_generator = AtomicFactGenerator("demos", self.lm)

            if verbose:
                topics = tqdm(topics)

            atomic_facts = []

            cache_file = "atomic_facts_cache.json"
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    atomic_facts_cache = json.load(f)
            else:
                atomic_facts_cache = {}
            
            for topic, gen in tqdm(zip(topics, generations)):
                if gen in atomic_facts_cache:
                    atomic_facts.append(atomic_facts_cache[gen])
                    continue

                # optionally, first detect if the response is abstained
                response_abstained = is_response_abstained(gen, self.abstain_detection_type)
                if response_abstained:
                    atomic_facts.append(None)
                    continue
                # continue only when the response is not abstained
                curr_afs, _ = self.af_generator.run(gen)
                curr_afs = [fact for _, facts in curr_afs for fact in facts]
                if len(curr_afs)==0:
                    atomic_facts.append(None)
                else:
                    atomic_facts.append(curr_afs)
                    atomic_facts_cache[gen] = curr_afs
                    lock = FileLock(cache_file)
                    with lock:
                        with open(cache_file, 'w') as f:
                            json.dump(atomic_facts_cache, f)

            assert len(atomic_facts)==len(topics)

        respond_ratio = np.mean([facts is not None for facts in atomic_facts])

        scores = []
        init_scores = []
        decisions = []
        for topic, generation, facts, input_index in tqdm(zip(topics, generations, atomic_facts, input_indexs), total=len(topics)):
            if start_idx is not None:
                if input_index < start_idx:
                    continue
            if end_idx is not None:
                if input_index > end_idx:
                    continue
            if facts is None:
                decisions.append(None)
            else:
                decision = self._get_score(topic, generation, facts, knowledge_source)
                #print(decision)
                score = np.mean([d["is_supported"] for d in decision])
                
                if gamma:
                    init_scores.append(score)
                    penalty = 1.0 if len(facts)>gamma else np.exp(1-gamma/len(facts))
                    score = penalty * score
                
                decisions.append(decision)
                scores.append(score)
                if len(scores) % 10 == 0:
                    self.save_cache()
            
            result_dict = {"prompt": topic, "response":generation, "score": score, "facts": facts, "decisions": decision}

            output_path = os.path.join(output_dir, f"result_{input_index}.json")
            
            with open(output_path, 'w') as f:
                json.dump(result_dict, f, indent=4) 

        self.save_cache()

        out = {"score": np.mean(scores),
               "respond_ratio": respond_ratio,
               "decisions": decisions,
               "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None])}

        if gamma:
            out["init_score"] = np.mean(init_scores)
        
        return out

    def _get_score(self, topic, generation, atomic_facts, knowledge_source, cost_estimate=None):
        decisions = []
        prompt_batch = []
        passages_batch = []
        for atom in atomic_facts:
            atom = atom.strip()
            max_steps = 5
            import wikipedia
            try:
                candid_topics = wikipedia.search(atom)[:max_steps]
            except Exception as e:
                candid_topics = []
                print("Encounter the following error when calling wikipedia: ", e)
            passages = []
            for step_id in range(max_steps):
                try:
                    passages = self.retrieval[knowledge_source].get_passages(candid_topics[step_id], atom, k=5)
                    if len(passages) > 0:
                        break
                except:
                    pass
            passages_batch.append(passages)
            #passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=5)
            definition = "Answer the question about {} based on the given context.\n\n".format(topic)
            context = ""
            for psg_idx, psg in enumerate(reversed(passages)):
                context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
            definition += context.strip()
            if not definition[-1] in string.punctuation:
                definition += "."
            prompt = "{}\n\nInput: {} True or False?\nOutput:".format(definition.strip(), atom.strip())
            prompt_batch.append(prompt)
        
        output_batch = self.lm.generate_batched(prompt_batch)

        for idx in range(len(atomic_facts)):
            generated_answer = output_batch[idx].lower()
            atom = atomic_facts[idx]

            if "true" in generated_answer or "false" in generated_answer:
                if "true" in generated_answer and "false" not in generated_answer:
                    is_supported = True
                elif "false" in generated_answer and "true" not in generated_answer:
                    is_supported = False
                else:
                    is_supported = generated_answer.index("true") > generated_answer.index("false")
            else:
                is_supported = all([keyword not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"]])

            if is_supported and "npm" in self.model_name:
                npprob = self.npm[knowledge_source].get_probabilty(topic, atom)
                is_supported = npprob > 0.3

            decisions.append({"atom": atom, "is_supported": is_supported, "retrieval_results": passages_batch[idx]})

        return decisions

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file',
                        type=str,)
    parser.add_argument('--col_name',
                        type=str,)
    parser.add_argument('--model_name',
                        type=str,
                        default="retrieval+ChatGPT")
    parser.add_argument('--gamma',
                        type=int,
                        default=10,
                        help="hyperparameter for length penalty")

    parser.add_argument('--openai_key',
                        type=str,
                        default="api.key")
    parser.add_argument('--data_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--model_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--cache_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--knowledge_source',
                        type=str,
                        default=None)


    parser.add_argument('--cost_estimate',
                        type=str,
                        default="consider_cache",
                        choices=["consider_cache", "ignore_cache"])
    parser.add_argument('--abstain_detection_type',
                        type=str,
                        default=None,
                        choices=["perplexity_ai", "generic", "none"])
    parser.add_argument('--use_atomic_facts',
                        action="store_true")
    parser.add_argument('--verbose',
                        action="store_true",
                        help="for printing out the progress bar")    
    parser.add_argument('--print_rate_limit_error',
                        action="store_true",
                        help="for printing out rate limit error when using OpenAI keys")
    parser.add_argument('--n_samples',
                        type=int,
                        default=None)
    parser.add_argument('--start_idx',
                      type=int, default=0)
    parser.add_argument('--end_idx',
                      type=int, default=9999999)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.ERROR if args.print_rate_limit_error else logging.CRITICAL)

    fs = FactScorer(model_name=args.model_name,
                    data_dir=args.data_dir,
                    cache_dir=args.cache_dir,
                    cost_estimate=args.cost_estimate,
                    abstain_detection_type=args.abstain_detection_type)

    global output_dir
    output_dir = args.input_file.replace(".csv", "")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_df = pd.read_csv(args.input_file)
    print(f"len dataset: {len(input_df)}")
    topics = list(input_df['user_prompts'].apply( lambda x: str(x)))
    generations = list(input_df[args.col_name].apply(lambda x: str(x)))
    # topics = ["why is democracy considered to be a desirable state or condition?"]
    # generations = ["Democracy is considered desirable for several reasons. Democracy is based on the principle of representing all citizens of a society. Every person's opinion matters and everyone has the right to express their view."]
    input_indexs = [tmp_idx for tmp_idx in range(len(topics))]

    out = fs.get_score(topics=topics,
                       generations=generations,
                       gamma=args.gamma,
                       atomic_facts=None,
                       knowledge_source=args.knowledge_source,
                       verbose=args.verbose,
                       input_indexs=input_indexs,
                       start_idx=args.start_idx,
                       end_idx=args.end_idx)
    logging.critical("FActScore = %.1f%%" % (100*out["score"]))
    if "init_score" in out:
        logging.critical("FActScore w/o length penalty = %.1f%%" % (100*out["init_score"]))
    logging.critical("Respond ratio = %.1f%%" % (100*out["respond_ratio"]))
    logging.critical("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))

    # Save out as a json file
    # with open("_factscore_output.json", 'w') as f:
    #     f.write(json.dumps(out) + "\n")

