
# NLP Final Project: Fine-tuning for Increased Extraversion in LLMs

This repo contains the codes for this project. 

## 

- Data cleaning: `data-cleaning.ipynb`
- Fine-tuning code: `finetuned-model` (not finetune_models, which is part of the classification implementation)
  This consists of two folders `empty-instr` and `please-instr` that include the models finetuned with empty instruction and please instruction datasets respectively.
- Responses and Result files: `responses-training-test`, `responses-baseline`, `results`, `results_baseline`
- Hyperparameter Tuning Results: `hyperparameter-tuning`
  This consists of all responses from the hyperparameter tuning trials and their prediction scores
- Average Score Calculator: `avg_score`, a Python script to average the extroversion trait scores for the responses. 
- Base model responses generation:  `base-eval.py`
- personality-prediction model: most of the remaining files are from the model linked (https://github.com/yashsmehta/personality-prediction)
