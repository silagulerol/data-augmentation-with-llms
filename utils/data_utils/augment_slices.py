"""
This script uses openai to augment the dataset with
samples for different few-shot domains
"""
import os, gc, torch, openai, pickle, json
import numpy as np
from . import eda_utils
from collections import Counter

pjoin = os.path.join

class GPTJChoice:
    def __init__(self, text):
        self.text = text

def load_dataset_slices(data_root, data_name):
    with open(pjoin(data_root, data_name, "full", "data_full_suite.pkl"), "rb") as f:
        return pickle.load(f)

def openai_complete(
    prompt,
    n,
    engine,
    temp,
    top_p,
    max_tokens=32,
    stop="\n",
    frequency_penalty=0,
    logprobs=None,
):
    completion = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=max_tokens,
        n=n,
        stop=stop,
        temperature=temp,
        top_p=1 if not top_p else top_p,
        frequency_penalty=frequency_penalty,
        logprobs=logprobs,
    )
    return completion.choices

def upsample_domain(prompt, n):
    lines = prompt.strip().splitlines()
    upsampled = lines * (n // len(lines))
    upsampled.extend(lines[: (n % len(lines))])
    return upsampled

def eda_domain(prompt, n):
    lines = prompt.strip().splitlines()
    k = len(lines)
    augmented = []
    for line in lines:
        if not line:
            continue
        generated = eda_utils.eda(
            sentence=line,
            alpha_sr=0.05,
            alpha_ri=0.05,
            alpha_rs=0.05,
            p_rd=0.05,
            num_aug=n // k,
        )
        augmented.extend(generated)
    return augmented

def regenerate(input_prompt, n_empty, engine, temp, top_p):
    new_lines = []
    while n_empty > 0:
        print(f"Saw {n_empty} empty line(s). GPT3ing again...")
        curr_lines = openai_complete(
            prompt=input_prompt,
            n=n_empty,
            engine=engine,
            temp=temp,
            top_p=top_p,
        )
        curr_lines = [r.text.strip() for r in curr_lines]
        n_empty = curr_lines.count("")
        new_lines.extend([t for t in curr_lines if t])
        if n_empty == 0:
            return new_lines

def augment_domain(
    dataset_slices,
    val_domain,
    data_save_path,
    id2name,
    num_ex=10,
    n_max=128,
    engine=None,
    temp=None,
    model=None,
    tokenizer=None,
    top_k=False,
    top_p=False,
    mode="upsample",
    mt_dict=None,
):
    if len(dataset_slices[val_domain]["M"]["train"]["intent"]) == 0:
        data_path = os.path.join(os.path.dirname(data_save_path), "dataset.pkl")
        with open(data_path, "rb") as f:
            dataset = pickle.load(f)
        num_synthetic = int(np.median(list(Counter(dataset["train"]["intent"]).values())))
    else:
        counts = Counter(dataset_slices[val_domain]["M"]["train"]["intent"])
        num_synthetic = int(np.median(list(counts.values())))

    f_train_lines = dataset_slices[val_domain]["F"]["train"]["text"]
    f_train_labels = dataset_slices[val_domain]["F"]["train"]["intent"]

    f_gen_lines, f_gen_labels = [], []
    for idx in range(0, len(f_train_lines), num_ex):
        prompt_lines = f_train_lines[idx : idx + num_ex]
        input_prompt = "\n".join([f"Example {i+1}: {t}" for i, t in enumerate(prompt_lines)])
        input_prompt += f"\nExample {num_ex+1}:"
        seed_intent = id2name[f"{f_train_labels[idx : idx + num_ex][0]}"]
        print(f"Seed intent: {seed_intent}")
        input_prompt = f"The following sentences belong to the same category '{seed_intent}':\n" + input_prompt

        if mode == "upsample":
            print("Upsampling...")
            generated_lines = upsample_domain(input_prompt, num_synthetic)
        elif mode == "eda":
            print("EDAing...")
            generated_lines = eda_domain(input_prompt, num_synthetic)
        else:
            print("GPT3ing...")
            if num_synthetic <= n_max:
                generated_lines = openai_complete(
                    prompt=input_prompt,
                    n=num_synthetic,
                    engine=engine,
                    temp=temp,
                    top_p=top_p,
                )
                generated_lines = [r.message["content"].strip() for r in generated_lines]
            else:
                generated_lines = []
                for _ in range(num_synthetic // n_max):
                    _c = openai_complete(
                        prompt=input_prompt,
                        n=n_max,
                        engine=engine,
                        temp=temp,
                        top_p=top_p,
                    )
                    generated_lines.extend([r.text.strip() for r in _c])
                _c = openai_complete(
                    prompt=input_prompt,
                    n=num_synthetic % n_max,
                    engine=engine,
                    temp=temp,
                    top_p=top_p,
                )
                generated_lines.extend([r.text.strip() for r in _c])

            n_empty = generated_lines.count("")
            if n_empty > 0:
                generated_lines = [t for t in generated_lines if t]
                if n_empty <= n_max:
                    generated_lines.extend(regenerate(input_prompt, n_empty, engine, temp, top_p))
                else:
                    for _ in range(n_empty // n_max):
                        generated_lines.extend(regenerate(input_prompt, n_max, engine, temp, top_p))
                    generated_lines.extend(regenerate(input_prompt, n_empty % n_max, engine, temp, top_p))

            assert len(generated_lines) == num_synthetic

        f_gen_lines.extend(generated_lines)
        f_gen_labels.extend([f_train_labels[idx]] * len(generated_lines))

    attr_name = mode if engine is None else f"{engine}_{temp}"
    dataset_slices[val_domain]["F"][attr_name] = {"text": f_gen_lines, "intent": f_gen_labels}
    write_pickle(data_save_path, dataset_slices)

def write_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def upsample_loop(dataset_slices, domains, data_save_path, id2name):
    for val_domain in domains:
        if "upsample" in dataset_slices[val_domain]["F"]:
            print(f"upsample for {val_domain} already exists")
            continue
        print(f"Augmenting for domain: {val_domain}")
        augment_domain(dataset_slices, val_domain, data_save_path, id2name)

def eda_loop(dataset_slices, domains, data_save_path, id2name):
    for val_domain in domains:
        if "eda" in dataset_slices[val_domain]["F"]:
            print(f"eda for {val_domain} already exists")
            continue
        print(f"Augmenting for domain: {val_domain}")
        augment_domain(dataset_slices, val_domain, data_save_path, id2name, mode="eda")

def gpt3_loop(dataset_slices, domains, ds_config, data_save_path, id2name, top_p):
    engine = "text-davinci-003"
    for temp in [1.0]:
        print(f"Engine: {engine} | Temp: {temp}")
        for val_domain in domains:
            if f"{engine}_{temp}" in dataset_slices[val_domain]["F"]:
                print(f"{engine}_{temp} for {val_domain} already exists")
                continue
            print(f"Augmenting for domain: {val_domain}")
            augment_domain(
                dataset_slices,
                val_domain,
                data_save_path,
                id2name,
                num_ex=ds_config.num_examples,
                n_max=ds_config.gpt3_batch_size,
                engine=engine,
                temp=temp,
                top_p=top_p,
                mode="gpt3",
            )


def augment_slices(data_root, ds_config, modes=["upsample", "gpt3", "eda"], top_k=False, top_p=False):
    dataset_slices = load_dataset_slices(data_root, ds_config.data_name)
    DOMAINS = ds_config.domain_to_intent.keys()
    data_save_path = pjoin(data_root, ds_config.data_name, "full", "data_full_suite.pkl")
    id2name = json.load(open(pjoin(data_root, ds_config.data_name, "id2name.json")))

    for mode in modes:
        if mode == "upsample":
            upsample_loop(dataset_slices, DOMAINS, data_save_path, id2name)
        elif mode == "gpt3":
            gpt3_loop(dataset_slices, DOMAINS, ds_config, data_save_path, id2name, top_p)
        elif mode == "eda":
            eda_loop(dataset_slices, DOMAINS, data_save_path, id2name)

    return dataset_slices
