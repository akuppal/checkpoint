"""
Mini reproduction of "Order Matters: Rethinking Prompt Construction in ICL" (arXiv 2511.09700)
ONE case: AG News, ONE demo set (k=8), P random orderings, N_test balanced test examples.
Output: accuracy per ordering + order sensitivity (std of accuracy across orderings).
Paper reference (GPT-5-nano, AG News, avg over 10 demo sets): order sensitivity ~= 0.0379

Usage:
  export OPENAI_API_KEY=sk-...
  python icl_order_agnews.py --train train.csv --test test.csv --model gpt-5-nano
  python icl_order_agnews.py --train train.csv --test test.csv --mock   # no API, pipeline check
"""
import argparse, json, os, random, re, statistics as st
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

LABELS = ["World", "Sports", "Business", "Sci/Tech"]  # AG News classes 0..3


def load(path):
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    if "text" in cols and "label" in cols:  # HuggingFace export
        out = pd.DataFrame({"text": df[cols["text"]], "label": df[cols["label"]].astype(int)})
    else:  # Kaggle: Class Index, Title, Description
        ci = cols.get("class index") or df.columns[0]
        ti = cols.get("title") or df.columns[1]
        de = cols.get("description") or df.columns[2]
        out = pd.DataFrame({"text": df[ti].astype(str) + ". " + df[de].astype(str),
                            "label": df[ci].astype(int) - 1})
    out["text"] = out["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    out["name"] = out["label"].map(lambda i: LABELS[i])
    return out


def balanced_sample(df, n, rng):
    per = n // 4
    parts = [df[df.label == c].sample(per, random_state=rng.randint(0, 10**6)) for c in range(4)]
    return pd.concat(parts).sample(frac=1, random_state=rng.randint(0, 10**6)).reset_index(drop=True)


def default_order(demos):
    # paper: group by label, labels alphabetical, examples alphabetical within label
    return sorted(demos, key=lambda d: (d["name"], d["text"]))


def build_prompt(demos, text):
    head = ("Classify each news article into one of: World, Sports, Business, Sci/Tech.\n"
            "Reply with ONLY the label for the final article. Do not repeat the examples "
            "and do not write anything else.\n\n")
    shots = "".join(f"Article: {d['text']}\nLabel: {d['name']}\n\n" for d in demos)
    return head + shots + f"Article: {text}\nLabel:"


def parse(resp):
    r = resp.lower()
    hits = [(r.rfind(l.lower()), l) for l in LABELS if l.lower() in r]
    return max(hits)[1] if hits else None


def make_caller(model, mock):
    if mock:
        return lambda p: random.choice(LABELS)
    from openai import OpenAI
    client = OpenAI()

    def call(prompt):
        for _ in range(5):
            try:
                r = client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}],
                    reasoning_effort="minimal", max_completion_tokens=200)
                return r.choices[0].message.content or ""
            except Exception as e:
                err = e
        print("call failed:", err)
        return ""
    return call


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--train", required=True)
    a.add_argument("--test", required=True)
    a.add_argument("--model", default="gpt-5-nano")
    a.add_argument("--k", type=int, default=8)
    a.add_argument("--perms", type=int, default=10)
    a.add_argument("--n_test", type=int, default=500)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--workers", type=int, default=16)
    a.add_argument("--mock", action="store_true")
    a.add_argument("--out", default="result.json")
    a = a.parse_args()

    rng = random.Random(a.seed)
    train, test = load(a.train), load(a.test)
    test = balanced_sample(test, a.n_test, rng)

    # ONE demo set: k/4 examples per class
    demo_df = pd.concat([train[train.label == c].sample(a.k // 4, random_state=rng.randint(0, 10**6))
                         for c in range(4)])
    base = default_order(demo_df.to_dict("records"))
    orders = []
    for _ in range(a.perms):
        o = base[:]
        rng.shuffle(o)
        orders.append(o)

    call = make_caller(a.model, a.mock)
    accs = []
    for i, o in enumerate(orders):
        prompts = [build_prompt(o, t) for t in test.text]
        with ThreadPoolExecutor(a.workers) as ex:
            raws = list(ex.map(call, prompts))
        preds = [parse(x) for x in raws]
        if i == 0:
            for r, n in list(zip(raws, test.name))[:6]:
                print(f"  DEBUG raw={r!r}  gold={n}")
        acc = sum(p == n for p, n in zip(preds, test.name)) / len(test)
        accs.append(acc)
        print(f"perm {i}: acc={acc:.4f}  (unparsed={sum(p is None for p in preds)})")

    sens = st.pstdev(accs) if len(accs) > 1 else 0.0
    print(f"\nmean acc={st.mean(accs):.4f}  order sensitivity (std)={sens:.4f}  [paper GPT-5-nano avg: 0.0379]")
    json.dump({"model": a.model, "accs": accs, "mean": st.mean(accs), "order_sensitivity_std": sens,
               "demo_set": base}, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()