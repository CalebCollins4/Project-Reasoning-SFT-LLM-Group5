import json, re, sys, random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import MiniBatchKMeans

MAXTOK = int(sys.argv[1]) if len(sys.argv) > 1 else 6044
K, TARGET, REP_MAX, LOW_PCT, SEED = 100, 15000, 0.05, 0.30, 42
OVERTHINK_DROP = 0.15
SRC = "source_data/data/acereason11_100k.json"
FEAT = "audit/features.jsonl"
OUT = "LLaMA-Factory/data/advanced15k_v5.json"
IDS = "audit/advanced15k_v5_ids.json"
REPORT = "audit/advanced15k_v5_report.json"

feats = [json.loads(l) for l in open(FEAT)]
print("features:", len(feats), flush=True)
data = json.load(open(SRC, encoding="utf-8"))
report = {"maxtok": MAXTOK, "clusters": K, "target": TARGET, "rep_max": REP_MAX,
          "drop_shortest_fraction_per_cluster": LOW_PCT, "overthink_drop_fraction": OVERTHINK_DROP, "seed": SEED}

cand = [f for f in feats if (not f["code"]) and f["boxed"] and f["tok"] <= MAXTOK
        and f["rep"] < REP_MAX and f["rep_lines"] == 0 and f["tail"] <= 400]
report["after_basic_filters"] = len(cand)
print("after basic filters:", len(cand), flush=True)

seen, uniq = set(), []
for f in cand:
    key = data[f["i"]]["instruction"].strip()
    if key in seen:
        continue
    seen.add(key)
    uniq.append(f)
cand = uniq
report["after_dedup"] = len(cand)
print("after dedup:", len(cand), flush=True)

pat = re.compile(r"\b(Wait|Hmm|Alternatively)\b")
dens = np.array([len(pat.findall(data[f["i"]]["output"])) / (f["tok"] / 1000.0) for f in cand])
cut = float(np.quantile(dens, 1 - OVERTHINK_DROP))
print("self-doubt markers per 1K tokens: median", round(float(np.median(dens)), 2), "| cut at", round(cut, 2), flush=True)
cand = [f for f, d in zip(cand, dens) if d <= cut]
report["after_overthink_filter"] = len(cand)
print("after overthink filter:", len(cand), flush=True)
if len(cand) < TARGET:
    sys.exit("Not enough candidates; raise MAXTOK or lower OVERTHINK_DROP")

texts = [data[f["i"]]["instruction"] for f in cand]
tf = TfidfVectorizer(max_features=30000, stop_words="english", sublinear_tf=True, min_df=3).fit_transform(texts)
X = Normalizer().fit_transform(TruncatedSVD(128, random_state=SEED).fit_transform(tf))
labels = MiniBatchKMeans(n_clusters=K, random_state=SEED, n_init=3, batch_size=4096).fit_predict(X)
sizes = np.bincount(labels, minlength=K)
print("cluster sizes min/median/max:", sizes.min(), int(np.median(sizes)), sizes.max(), flush=True)

w = np.sqrt(sizes.astype(float))
quota = np.zeros(K, dtype=int)
remaining = TARGET
while remaining > 0:
    active = [c for c in range(K) if quota[c] < sizes[c]]
    if not active:
        break
    wsum = sum(w[c] for c in active)
    gave = 0
    for c in active:
        a = min(int(remaining * w[c] / wsum), sizes[c] - quota[c])
        quota[c] += a
        gave += a
    if gave == 0:
        for c in sorted(active, key=lambda c: sizes[c] - quota[c], reverse=True):
            if gave >= remaining:
                break
            quota[c] += 1
            gave += 1
    remaining -= gave
print("quota min/max:", quota.min(), quota.max(), "sum:", quota.sum(), flush=True)

toks = np.array([f["tok"] for f in cand])
chosen = []
for c in range(K):
    q = int(quota[c])
    if q == 0:
        continue
    idx = np.where(labels == c)[0]
    idx = idx[np.argsort(toks[idx], kind="stable")]
    pool = idx[int(len(idx) * LOW_PCT):]
    if len(pool) < q:
        pool = idx
    pos = np.linspace(0, len(pool) - 1, q).round().astype(int)
    chosen.extend(pool[pos].tolist())

assert len(chosen) == TARGET and len(set(chosen)) == TARGET, (len(chosen), len(set(chosen)))
random.Random(SEED).shuffle(chosen)
sel = [cand[j] for j in chosen]

out = []
for f in sel:
    item = data[f["i"]]
    out.append({
        "instruction": item["instruction"],
        "input": item.get("input", ""),
        "output": item["output"],
        "system": item.get("system") or "Please reason step by step, and put your final answer within \\boxed{}.",
    })
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
json.dump([f["i"] for f in sel], open(IDS, "w"))

st = np.array([f["tok"] for f in sel])
report.update({"selected": len(out), "tokens_median": int(np.median(st)), "tokens_mean": int(st.mean()),
               "tokens_max": int(st.max()), "tokens_total_millions": round(float(st.sum()) / 1e6, 1),
               "candidate_tokens_median": int(np.median(toks)), "clusters_used": int((quota > 0).sum())})
json.dump(report, open(REPORT, "w"), indent=2)
print("\n=== SELECTED 15K (v5) ===")
print(json.dumps(report, indent=2))
print("\n--- sample prompts ---")
for c in range(0, K, 17):
    for j in chosen:
        if labels[j] == c:
            print(f"[cluster {c}]", data[cand[j]["i"]]["instruction"][:110].replace("\n", " "))
            break
