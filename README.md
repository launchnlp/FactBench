# 🔎 FactBench: A Dynamic Benchmark for In-the-Wild Language Model Factuality Evaluation
<p align="center">
| <a href="https://huggingface.co/spaces/launch/factbench"><b>🏆 Leaderboard</b></a> | <a href="https://arxiv.org/abs/2410.22257"><b>📑 arXiv</b></a> | <a href="https://x.com/FarimaFB/status/1851658704829272566"><b>🐦 Twitter/X</b></a> |
</p>

This is the official code repo of our paper ["FactBench: A Dynamic Benchmark for In-the-Wild Language Model Factuality Evaluation"](https://huggingface.co/spaces/launch/factbench).
This repository contains:

1. **FactBench**: A new dynamic factuality benchmark grounded in the real-world usage of LMs. All related codes for constructing the benchmark is under ``./FactBench`` folder.
2. **VERIFY**: A factuality evaluation pipeline that considers the verifiability of generated content and categorizes units into *supported*, *unsupported*, or *undecidable* according to retrieval results. Codes available under ``./VERIFY`` folder.
3. **Baselines** ([FActScore](https://github.com/shmsw25/FActScore), [SAFE](https://github.com/google-deepmind/long-form-factuality), [Factcheck-GPT](https://github.com/yuxiaw/Factcheck-GPT)): Related previous works that serve as our baselines. All baselines are accelerated and adapted to our framework. Codes available under ``./baselines`` folder.


## Pipeline Diagram
<p align="center">
<img src="assets/factEvalSteps.jpg" width=100%>
</p>

## Accessing the Repository

First, clone our GitHub repository and navigate to the newly created folder:

```bash
git clone https://github.com/launchnlp/FactBench.git
cd FactBench
```

## Environment Setup and Factuality Evaluation 

### If running VERIFY (Our Factuality Evaluation Pipeline):
1. Install all requirements & dependencies:
```bash
pip install -r requirements.txt
```

2. Put **FactBench** data under:
```bash
./VERIFY/data/lmsys_data/final_dataset/
```

3. Run **VERIFY** pipeline:
```bash
cd VERIFY
python factuality_evaluation.py --backbone_llm "Llama-3-70B-Instruct" --cache_dir "./cache/" --tier_number 1 --model_name "gpt4-o" 
```

4. You should be about to find evaluation results under:
```bash
./VERIFY/data/lmsys_data/benchmarking/BenchCurator
```

## Add your favorite models to Leaderboard
Please consider raising issues [here](https://github.com/launchnlp/FactBench/issues/new) and mention the name of your new models!

## Citation

If you find our work for your research, please cite our [paper](https://arxiv.org/abs/2410.22257):
```bibtex
@misc{bayat2024factbenchdynamicbenchmarkinthewild,
      title={FactBench: A Dynamic Benchmark for In-the-Wild Language Model Factuality Evaluation}, 
      author={Farima Fatahi Bayat and Lechen Zhang and Sheza Munir and Lu Wang},
      year={2024},
      eprint={2410.22257},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.22257}, 
}
```