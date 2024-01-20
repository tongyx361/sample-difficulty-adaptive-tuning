# Sample-Difficulty-Adaptive Supervised Fine-Tuning for Mathematical Reasoning

Code from project *Sample-Difficulty-Adaptive Supervised Fine-Tuning for Mathematical Reasoning* at course ANN-23F@THU-CST.
Report can be found [here](./report.pdf).

## Installation

We use `flash-attn` to accelerate the training and inference, which has the following requirements:

> CUDA 11.6 and above.
> PyTorch 1.12 and above.
> Linux. Might work for Windows starting v2.3.2 (we've seen a few positive reports) but Windows compilation still requires more testing. If you have ideas on how to set up prebuilt CUDA wheels for Windows, please reach out via Github issue.

```shell
conda create -n sdat python=3.11
conda activate sdat
```

Taking CUDA 11.7 for example, first install PyTorch:

```shell
# https://pytorch.org/get-started/previous-versions/#v201
# CUDA 11.7
conda install pytorch==2.0.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Then install other packages:

```shell
pip install -r requirements.txt
```

Finally install `flash-attn` at last for less building errors:

```shell
# https://github.com/Dao-AILab/flash-attention
pip install flash-attn --no-build-isolation
```

## Dataset Loading and Analysis

```python
from datasets import load_dataset
dp_info_ds = load_dataset("tongyx361/MathInstruct-Core-DifficultyAware")
```

For more details, please check `load-and-analyze-dataset.ipynb`, in which you can also reproduce Figure 1 by just clicking "Run All Cells".

## Training

To reproduce training, please first set values for base and weight in `train.sh`, for example:

```shell
adap_base=1
adap_factor=2
```

Then just run

```shell
bash train.sh
```

Basically, this script implements the main method, i.e. sample-difficulty-adaptive tuning, suggested in the report and train the model.

For more details, please check `train.sh` and `train.py`.

## Evaluation

To reproduce evaluation, please run the following command with different values for `model`, `stem_flan_type` and `eval_dataset`:

```shell
# For prompting, set `stem_flan_type` to one of
# - "" (CoT)
# - "pot_prompt" (PoT)

# For dataset to evaluate on, set `eval_dataset` to one of
# ['gsm8k', 'svamp', 'math', 'numglue', 'deepmind', 'simuleq'] (open-eneded)
# ['aqua', 'sat', 'mmlu_mathematics', "mmlu_physics", "mmlu_chemistry", "mmlu_biology"] (mutilple-choice)

model="models/llemma-7b-math-instruct-diffi-aware-pre-b1-w2-bs128-gas1-lr2e-5-wd0-warmratio0.03-sched-cosine-epochs1-maxlen512"

python eval-model.py\
    --model_name_or_path "${model}"\
    --stem_flan_type "pot_prompt"\
    --eval_dataset "math"\
    --gpu_ids "4"\
    --eval_results_dir "eval-results"\
    --eval_datasets_dir "eval-datasets"
```

Or just set the values in `eval-model.sh` and run

```shell
bash eval-model.sh
```

Basically, this script will
1. load the model from `model`,
2. inference with `vllm` on `eval_dataset` from `eval_datasets_dir`([`eval-datasets`](./eval-datasets) by default) with prompting `stem_flan_type`,
3. extract and grade the answers,
3. save the results to `eval_results_dir`.

For more details, please check `eval-model.sh` and `eval-model.py`.

## Result Analysis

In directory [eval-results](./eval-results), we provide the detailed generations, answers of which have all been extracted and graded.

To reproduce result analysis, please check `analyze-eval-results.ipynb`, in which you can reproduce Figure 2+ by just clicking "Run All Cells".

Basically, this notebook will
1. load the evaluation results from `eval_results_dir`([`eval-results`](./eval-results) by default),
2. calculate various metrics,
3. visualize the results.

## FAQ

> Why are my reproduced results slightly different from reported in the report?

Different versions of packages like transformers, pytorch, etc. could cause negligible but non-zero performance differences.