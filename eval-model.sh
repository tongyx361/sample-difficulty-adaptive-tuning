#! /bin/bash
# For prompting, set `stem_flan_type` to one of
# - "" (CoT)
# - "pot_prompt" (PoT)

# For dataset to evaluate on, set `eval_dataset` to one of
# ['gsm8k', 'svamp', 'math', 'numglue', 'deepmind', 'simuleq'] (open-eneded)
# ['aqua', 'sat', 'mmlu_mathematics', "mmlu_physics", "mmlu_chemistry", "mmlu_biology"] (mutilple-choice)

# model="models/llemma-7b-math-instruct-diffi-aware-pre-b1-w2-bs128-gas1-lr2e-5-wd0-warmratio0.03-sched-cosine-epochs1-maxlen512"

model="/ssddata/tongyx/projects/deita-domain/DEITA-Domain/models/mistral-7b-math-instruct-core-hard0.80-math-cmp-rtlxprm-div1.00-tokens8388657-log2-23-bs64-gas1-lr5e-6-wd0-warmratio0.03-sched-cosine-epochs3-maxlen512"

python eval-model.py\
    --model_name_or_path "${model}"\
    --stem_flan_type "pot_prompt"\
    --eval_dataset "math"\
    --gpu_ids "4"\
    --eval_results_dir "eval-results"\
    --eval_datasets_dir "eval-datasets"
