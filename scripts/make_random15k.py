"""Build the Random15K subset: 15,000 examples sampled uniformly from acereason11_100k.

Run from the repository root:
    python scripts/make_random15k.py

Writes LLaMA-Factory/data/random15k.json (about 353 MB). The `random15k` entry in
LLaMA-Factory/data/dataset_info.json registers it for training.
Published copy: https://huggingface.co/datasets/TML-Project1/acereason-random15k
"""
import json
import os
import random

from huggingface_hub import hf_hub_download

SEED = 42
N = 15000

src = hf_hub_download(repo_id="redsgnaoh/acereason11_100k", repo_type="dataset",
                      filename="data/acereason11_100k.json",
                      local_dir=os.path.expanduser("~/data"))
data = json.load(open(src))

random.seed(SEED)
sub = random.sample(data, N)

out = os.path.join("LLaMA-Factory", "data", "random15k.json")
json.dump(sub, open(out, "w"))
print(len(data), "->", len(sub), "| keys:", list(sub[0].keys()), "| wrote", out)
