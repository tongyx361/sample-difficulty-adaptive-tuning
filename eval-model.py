# %%
import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from statistics import mean

from prompt_utils import get_examples, get_prompt
from tqdm import tqdm
from utils import (
    compare,
    delete_extra_zero,
    eval_dataset2ans_mode,
    extract_ans_considering_code,
    init_logging,
    process_question_with_flan_tag,
    recover_options,
)
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate on specified evaluation dataset(s) with specified prompt(s)"
    )
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs to use")
    parser.add_argument(
        "--temperature", type=float, default=0, help="Temperature for sampling"
    )
    parser.add_argument(
        "--model_max_len", type=int, default=1500, help="Maximum model length"
    )
    parser.add_argument(
        "--cpu_ratio_for_code_execution",
        type=float,
        default=0.8,
        help="CPU ratio for code execution",
    )
    parser.add_argument(
        "--stem_flan_type",
        type=str,
        default="",
        choices=["", "pot_prompt"],
        help="FLAN type for the STEM field",
    )
    parser.add_argument(
        "--eval_dataset", type=str, default="math", help="Dataset for evaluation"
    )
    parser.add_argument(
        "--eval_datasets_dir",
        type=str,
        default=None,
        help="Directory for evaluation datasets",
    )
    parser.add_argument(
        "--nshots", type=int, default=0, help="Number of shots for prompting"
    )
    parser.add_argument("--form", type=str, default="alpaca", help="Form of the prompt")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="TIGER-Lab/MAmmoTH-Coder-7B",
        help="Model name or path",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="Data type for the model"
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for randomness")
    parser.add_argument(
        "--trust_remote_code", action="store_true", help="Flag to trust remote code"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument(
        "--eval_results_dir",
        type=str,
        default=None,
        help="Directory for evaluation results",
    )
    parser.add_argument(
        "--project_home",
        type=str,
        default=os.environ.get("PROJECT_HOME", "."),
        help="Project home directory",
    )
    parser.add_argument(
        "--match_choice", default="self||A", choices=["self", "A", "self||A"], type=str
    )

    args = parser.parse_args()

    if args.eval_datasets_dir is None:
        args.eval_datasets_dir = os.path.join(args.project_home, "eval-datasets")
    if args.eval_results_dir is None:
        args.eval_results_dir = os.path.join(args.project_home, "eval-results")

    return args


args = parse_args()

init_logging()


def get_results_path(stem_flan_type: str):
    results_filename = args.eval_dataset
    prompt = None
    if stem_flan_type == "pot_prompt":
        prompt = "pot"
    else:
        prompt = "cot"
    results_filename += "-" + prompt
    results_filename += ".json"

    model_dirname = None
    if args.model_name_or_path.startswith("/"):
        model_dirname = os.path.basename(args.model_name_or_path)
    else:
        model_dirname = args.model_name_or_path.replace("/", "--")

    model_metrics_dir = os.path.join(args.eval_results_dir, model_dirname)
    results_dir = os.path.join(model_metrics_dir, "results")

    os.makedirs(results_dir, exist_ok=True)

    results_filepath = os.path.join(results_dir, results_filename)

    return results_filepath


results_filepath = get_results_path(args.stem_flan_type)

if os.path.exists(results_filepath):
    logging.info(f"Results file {results_filepath} already exists!")
    sys.exit(0)


logging.info(f"args.model_name_or_path = {args.model_name_or_path}")

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.info(
    f"""
    args.gpu_ids = {args.gpu_ids}
    os.environ["CUDA_VISIBLE_DEVICES"] = {os.environ["CUDA_VISIBLE_DEVICES"]}
    """
)


navail_cpu_cores = int(mp.cpu_count() * args.cpu_ratio_for_code_execution)

ans_mode = eval_dataset2ans_mode[args.eval_dataset]

# vllm

stop_tokens = [
    "Question:",
    "Question",
    "USER:",
    "USER",
    "ASSISTANT:",
    "ASSISTANT",
    "Instruction:",
    "Instruction",
    "Response:",
    "Response",
    "### Instruction",
]

if args.temperature <= 1e-5:
    args.temperature = 0
    args.num_samples = 1
    args.top_p = 1

sampling_params = SamplingParams(
    n=args.num_samples,
    temperature=args.temperature,
    top_p=args.top_p,
    max_tokens=args.model_max_len,
    stop=stop_tokens,
    skip_special_tokens=True,
)


def load_eval_dataset(dataset: str):
    """Load the evaluation dataset and return the questions and ground truths."""
    questions = []
    ground_truths = []
    decoder = json.JSONDecoder()

    if dataset == "aqua":
        with open(os.path.join(args.eval_datasets_dir, "AQuA/AQuA.json")) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "(" + "(".join(json_res["options"])
                choice = choice.replace("(", " (").replace(")", ") ")
                choice = "Answer Choices:" + choice
                questions.append(json_res["question"].strip() + "\n" + choice)
                ground_truths.append(json_res["correct"])
    elif dataset == "math":
        with open(os.path.join(args.eval_datasets_dir, "math/MATH.json"), "r") as f:
            loaded = json.load(f)
        for d in loaded:
            questions.append(d["question"])
            ground_truths.append(d["answer"])
    elif dataset == "gsm8k":
        with open(os.path.join(args.eval_datasets_dir, "gsm8k/gsm8k.jsonl")) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                ground_truths.append(
                    delete_extra_zero(
                        json_res["answer"].split("#### ")[-1].replace(",", "")
                    )
                )
    elif dataset == "svamp":
        with open(os.path.join(args.eval_datasets_dir, "SVAMP/SVAMP.json")) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                ground_truths.append(delete_extra_zero(a))
    elif "mmlu" in dataset:
        with open(
            os.path.join(args.eval_datasets_dir, f'mmlu/{dataset.split("_")[1]}.json')
        ) as f:
            json_data = json.load(f)
            for line in json_data:
                options = f'(A) {line["choices"][0]} (B) {line["choices"][1]} (C) {line["choices"][2]} (D) {line["choices"][3]}'
                q = line["question"] + "\n" + "Answer Choices: " + options
                a = ["A", "B", "C", "D"][line["answer"]]
                questions.append(q)
                ground_truths.append(a)
    elif dataset in ["numglue", "simuleq", "deepmind", "sat"]:
        with open(
            os.path.join(args.eval_datasets_dir, f"{dataset}/{dataset}.json")
        ) as f:
            json_data = json.load(f)
            for line in json_data:
                assert isinstance(line["question"], str) and isinstance(
                    line["question"], str
                ), line
                questions.append(line["question"])
                ground_truths.append(str(line["answer"]))
    else:
        raise ValueError("dataset is not properly defined ...")

    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)

    print(f"dataset = {dataset}")
    print(f"data size : {len(ground_truths)}")
    print(f"average num of words for each sample : {q_len_mean}")

    return questions, ground_truths


def batch_extract_ans_considering_code(outputs, executable=True):
    idx2ans, idx2err = {}, {}

    logging.info(f"len(outputs) = {len(outputs)}")
    # wrong meanings for initialization only
    old_ntimeout_outputs = -1
    ntimeout_outputs = len(outputs)
    idx2output = {idx: output for idx, output in enumerate(outputs)}
    timeout_idx2output = idx2output

    while ntimeout_outputs > 0 and ntimeout_outputs != old_ntimeout_outputs:
        logging.info("Extracting answers ...")
        with mp.Pool(processes=min(navail_cpu_cores, len(timeout_idx2output))) as pool:
            for idx, output in tqdm(timeout_idx2output.items()):
                idx2ans[idx], idx2err[idx] = extract_ans_considering_code(
                    pool, output, ans_mode, executable
                )
        logging.info("Extracting answers done!")
        old_ntimeout_outputs = ntimeout_outputs
        timeout_idx2output = {
            idx: output
            for idx, output in timeout_idx2output.items()
            if "multiprocessing.context.TimeoutError" in str(output)
        }
        ntimeout_outputs = len(timeout_idx2output)
        logging.info(f"ntimeout_outputs = {ntimeout_outputs}")
    logging.info(f"Converged with {ntimeout_outputs} samples timeout!")

    return list(idx2ans.values()), list(idx2err.values())


# %%
logging.info(f"Loading evaluation dataset: {args.eval_dataset}")
questions, ground_truths = load_eval_dataset(args.eval_dataset)
logging.info(
    f"""
    questions[0] = ({questions[0]})
    ground_truths[0] = ({ground_truths[0]})
    """
)

# Add prefix

questions = process_question_with_flan_tag(questions, args.stem_flan_type)
logging.info(f"questions[0] = ({questions[0]})")

# Add examplars

examplars = get_examples(args.eval_dataset, args.nshots, args.stem_flan_type)

prompt_no_input, prefix = get_prompt(examplars, args.form)

# Format

input_strs = [prompt_no_input + prefix.format(query=q) for q in questions]
logging.info(
    f"""
    len(input_strs) = {len(input_strs)}
    input_strs[0] = (\n{input_strs[0]}\n)
    """
)

# %%
llm = LLM(
    model=args.model_name_or_path,
    tokenizer=None,
    tensor_parallel_size=args.num_gpus,
    dtype=args.dtype,
    seed=args.seed,
    trust_remote_code=args.trust_remote_code,
    swap_space=80,
)

# %%
raw_outputs = llm.generate(input_strs, sampling_params)
logging.info(f"len(raw_outputs) = {len(raw_outputs)}")
logging.info(f"raw_outputs[0] = ({raw_outputs[0]})")

# %%
outputs = [output.outputs[0].text for output in raw_outputs]
logging.info(f"outputs[0] = {outputs[0]}")

# %%
# Collect the (question, output, answer, ground_truth)s and possibly the rerun questions
answers, errors = batch_extract_ans_considering_code(outputs)

# %%
# assert (
#     len(answers) == len(errors) == len(ground_truths) == len(questions) == len(outputs)
# )

# Anonimize the absolute paths
answers = [ans.replace(args.project_home, "/path/to/project/home") for ans in answers]
logging.info(f"answers[0] = ({answers[0]})")

ALL_CHOICES = ["A", "B", "C", "D", "E"]

results = [
    {"ans": ans, "gt": gt, "err": err, "question": q, "output": output}
    for ans, gt, err, q, output in zip(
        answers, ground_truths, errors, questions, outputs
    )
]

if ans_mode == "multi-choice":
    matching_indices = [
        idx
        for idx, ans in enumerate(answers)
        if ans not in ALL_CHOICES
        # If the answer is not an option at all.
    ]
    # while len(matching_indices) > 0:
    # only once
    if len(matching_indices) > 0:
        logging.info(
            f"""
                len(matching_indices) = {len(matching_indices)}
            """
        )
        # matching_indices = {matching_indices}
        if "self" in args.match_choice:
            logging.info(
                "Matching the final answer by LLM with the raw and the options!"
            )
            matching_input_strs = []
            prompt_no_input, prefix = get_prompt([], form=args.form)
            for idx in matching_indices:
                options = recover_options(questions[idx], combined=True)
                q = f"Please find the closest option to {answers[idx]}. The options are {options}"
                matching_input_str = prompt_no_input + prefix.format(query=q)
                matching_input_strs.append(matching_input_str)
            # Formulate the real prompt
            matching_raw_outputs = llm.generate(matching_input_strs, sampling_params)
            matching_outputs = [
                output.outputs[0].text for output in matching_raw_outputs
            ]
            matching_choices, matching_errs = batch_extract_ans_considering_code(
                matching_outputs
            )
            # matching_infos = list(zip(matching_indices, matching_choices, matching_errs))
            for i, choice in enumerate(matching_choices):
                if choice not in ALL_CHOICES:
                    if "A" in args.match_choice:
                        matching_choices[i] = "A"
                    matching_errs[i] = True
        elif args.match_choice == "A":
            logging.info("Default the option to A!!!")
            nmatchings = len(matching_indices)
            matching_choices = ["A"] * nmatchings
            matching_errs = [False] * nmatchings

        for idx, matching_choice, matching_err, matching_output in zip(
            matching_indices, matching_choices, matching_errs, matching_outputs
        ):
            raw_ans = results[idx]["ans"]
            err = results[idx]["err"]
            gt = results[idx]["gt"]
            q = results[idx]["question"]
            output = results[idx]["output"]
            if raw_ans in ALL_CHOICES:
                raise ValueError("raw_ans is an option!")
            if matching_choice not in ALL_CHOICES:
                logging.warning("Fail to match the answer!")
                continue

            results[idx] = {
                "ans": matching_choice,
                "err": err,
                "gt": gt,
                "raw_ans": raw_ans,
                "question": q,
                "output": output,
                "matching_err": matching_err,
                "matching_strategy": args.match_choice,
            }
        sample_matching_idx = matching_indices[0]
        logging.info(f"results[{sample_matching_idx}] = {results[sample_matching_idx]}")

# %%

for result in results:
    result["correct"] = compare(result["ans"], result["gt"])

# %%
with open(
    results_filepath,
    "w",
) as f:
    json.dump(results, f, indent=4)

# %%
nquestions = len(questions)
cnt_correct = sum([result["correct"] for result in results])
correct_rate = cnt_correct / nquestions

logging.info(
    f"""
    model        = {args.model_name_or_path}
    eval_dataset = {args.eval_dataset}
    prompting    = {"PoT" if "pot" in args.stem_flan_type else "CoT"}
    nshot        = {args.nshots}
    nquestions   = {nquestions}
    correct_rate = {correct_rate:.2%}
"""
)
