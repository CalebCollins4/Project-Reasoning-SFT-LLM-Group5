import json, shutil, os, time
d = "LLaMA-Factory/data"
assert os.path.exists(f"{d}/advanced15k_v5.json"), "advanced15k_v5.json not found - wait for the selection job"
info = json.load(open(f"{d}/dataset_info.json"))
assert "advanced15k_v5" not in info, "already registered"
shutil.copy(f"{d}/dataset_info.json", f"{d}/dataset_info.json.bak_v5_{int(time.time())}")
info["advanced15k_v5"] = {"file_name": "advanced15k_v5.json",
    "columns": {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}}
json.dump(info, open(f"{d}/dataset_info.json", "w"), indent=2, ensure_ascii=False)
print("registered advanced15k_v5; entries now:", len(info))
