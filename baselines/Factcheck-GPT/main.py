import sys
sys.path.append("./src")
import pandas as pd
from pipeline import check_document, check_documents
import argparse
import os
from tqdm import tqdm
import json
from utils.openaiAPI import Model

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_file',
                      type=str,)
  parser.add_argument('--col_name',
                      type=str,)
  parser.add_argument('--start_idx',
                      type=int, default=0)
  parser.add_argument('--end_idx',
                      type=int, default=9999999)
  parser.add_argument('--whose_key',
                      type=str, default="farima")
  args = parser.parse_args()

  output_dir = args.input_file.replace(".csv", "")
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  model = Model("openai:gpt-35-turbo", whose_key=args.whose_key)
  input_df = pd.read_csv(args.input_file)
  print(f"len dataset: {len(input_df)}")
  for idx in tqdm(input_df.index, desc=args.col_name):
      if idx < args.start_idx or idx > args.end_idx:
        continue
      row = input_df.iloc[idx]
      prompt = row['user_prompts']
      response = str(row[args.col_name])

      label, log = check_document(response, model = model)
      result_dict = {
        'prompt': prompt, 'response': response, "log": log.to_dict('records')
        }
      output_path = os.path.join(output_dir, f"{idx}.json")
      print(label)
      with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=4) 