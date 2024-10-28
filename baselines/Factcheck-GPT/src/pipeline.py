import os
import pandas as pd
from decompose import doc2sentences
from checkworthy import identify_checkworthiness
from retrieve import get_web_evidences_for_claim, generate_questions_in_batch
from verify import verify_document, verify_document_batched

def check_document(doc: str, model, num_retries: int=3):
    """input: a document
       output: factuality label and a DataFrame log (claim, evidence, checkworthy, verification)"""
    # split into claims or sents
    # sents = doc2sentences(doc)
    claims = doc2sentences(doc, mode="claims", model=model)
    checkworthy_labels = identify_checkworthiness(claims, model=model)

    # retrieve evidence for checkworthy claims
    raw_evidence = []
    evidence = []
    interested_idx, question_batch = [idx for idx in range(len(claims)) if checkworthy_labels[idx].lower() == "yes"], [None for _ in range(len(claims))]
    question_batch_sub = generate_questions_in_batch([claims[idx] for idx in interested_idx], model)
    for idx in range(len(interested_idx)):
        question_batch[interested_idx[idx]] = question_batch_sub[idx]
    for i, claim in enumerate(claims):
        evidences = None
        if checkworthy_labels[i].lower() == "yes":
            evidences = get_web_evidences_for_claim(claim, model=model, questions=question_batch[i])
            evids = [evid['text'] for evid in evidences['aggregated']]
        else:
            evids = []
        evidence.append(evids)
        raw_evidence.append(evidences)

    # based on the evidence and claim, verify true/false
    label, log = verify_document_batched(claims, evidence, model=model, num_retries=num_retries)
    log["checkworthy"] = checkworthy_labels
    log["raw_evidence"] = raw_evidence
    return label, log


def check_documents(filename = "./factcheck-GPT-benchmark.jsonl", 
                    model="gpt-3.5-turbo-0613", num_retries=3,
                    savepath = "./logs/"):
    
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    labels = []
    df = pd.read_json(filename, lines=True)
    # for test
    # df = df[:2]
    for i, row in df.iterrows():
        doc = row["response"]
        label, log = check_document(doc, model=model, num_retries=num_retries)
        log.to_json(os.path.join(savepath, f"{i}.jsonl"), lines=True, orient="records", encoding='utf-8')
        labels.append(label)
    return labels

    # if perform in batch step by step
    # responses = list(df["response"])
    # prompts = list(df["prompt"])
    # data = df[:10]

    # decompose and decontextualize
    # sentences, claims = [], []
    # for i, row in data.iterrows():
    #     sentences.append(doc2sentences(row["response"]))
    #     claims.append(doc2sentences(row["response"], mode="claims"))
    # data["chatgpt_sent"] = sentences
    # data["chatgpt_claims"] = claims

    # # checkworthy
    # checkworthy_predictions = []
    # for claim in claims:
    #     results = identify_checkworthiness(claim)
    #     checkworthy_predictions.append(results)
    # data["claim_checkworthy"] = checkworthy_predictions
    # data.to_json(savepath, lines=True, orient="records")

    # specify the checkworthy type given a sentence or claim
    # specify_checkworthiness_type("Is Preslav a professor in MBZUAI.")


    # retrieve evidence for claim: given a claim, return the most related five passages
    # claim = "Yuxia Wang is graduated from the Univerisity of Melbourne."
    # evidences = get_web_evidences_for_claim(claim)
    # evids = [evid['text'] for evid in evidences['aggregated']]