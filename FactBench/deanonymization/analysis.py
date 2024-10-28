import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the CSV file into a DataFrame
input_file_path_gpt4 = "../data/lmsys_data/lmsys_user_prompts_jaccard0.9_w_factually_w_topics_w_model_v1.1_prompts_evaluated_gpt4.csv"
input_file_path_llama3 = "../data/lmsys_data/lmsys_user_prompts_jaccard0.9_w_factually_w_topics_w_model_v1.1_prompts_evaluated_llama3.csv"
data = pd.read_csv(input_file_path)
scores = data["scores"].values.tolist()

# List of criteria
criteria = ['Clarity', 'Generalizability', 'Relevance', 'Actionability', 'Feasibility']
print("length scores: ", len(scores))
# Plot the distribution for each criterion
for criterion in criteria:
    data_c = []
    for idx, score in enumerate(scores):
        # print(f"id: {idx}, score: {score}")
        start_index = score.find('{')
        end_index = score.find('}', start_index) + 1
        score_str = score[start_index:end_index]
        try:
            score_j = json.loads(score_str)
            # print("json.loads(score): ", score_j)
            data_c.append(score_j[criterion])
        except:
            pass
            # print(f"Error id: {idx}, score: {score}")
    # print(data_c)
    plt.hist(np.array(data_c), bins=range(min(data_c), max(data_c)+2), edgecolor='black', rwidth=0.8)
    plt.title(f'Distribution of {criterion}')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    # plt.show()
    plt.savefig(f'figures/{criterion}_distribution.png')  # Save the plot as an image file 
 
