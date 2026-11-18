"""Retry contrast over accepted application packages, like for like.
python3 -I retry_like_for_like.py -> out/retry_like_for_like.json
Imports the pricing and grouping from prod.py (reads the private event stream, no network).
Compares accepted packages with a retry call to accepted packages that made calls but no retry,
(a) as the paper's original contrast, (b) restricted to packages that ran at least one judge call
(the rewrite-and-judge loop, the only place a retry can occur), with a 10,000-resample bootstrap
(seed 2027) of the ratio of medians. Also the retry role's own dollars per package and the call
medians of both groups."""
import io, json, random, sys, statistics as st, contextlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
with contextlib.redirect_stdout(io.StringIO()):
    import prod
op_usd, op_calls, op_role_calls, op_role_usd = prod.op_usd, prod.op_calls, prod.op_role_calls, prod.op_role_usd
acc = prod.pk_acc
ret = [o for o in acc if op_role_calls[o]["retry"]]
nor = [o for o in acc if not op_role_calls[o]["retry"] and op_calls[o]]
def summ(ops):
    return {"n": len(ops), "median_usd": round(st.median(op_usd[o] for o in ops), 4),
            "median_calls": st.median(op_calls[o] for o in ops),
            "median_usd_excluding_retry": round(st.median(op_usd[o] - op_role_usd[o]["retry"] for o in ops), 4)}
def ratio_ci(a, b):
    random.seed(2027)
    xa = [op_usd[o] for o in a]; xb = [op_usd[o] for o in b]
    rs = sorted(st.median(random.choices(xa, k=len(xa))) / st.median(random.choices(xb, k=len(xb))) for _ in range(10000))
    return [round(rs[250], 2), round(rs[9749], 2)]
out = {"all_with_calls": {"retry": summ(ret), "no_retry": summ(nor)}}
out["all_with_calls"]["ratio"] = round(out["all_with_calls"]["retry"]["median_usd"] / out["all_with_calls"]["no_retry"]["median_usd"], 2)
out["all_with_calls"]["ratio_ci95_from_prod_py"] = prod.r3["retry_vs_noretry_ratio_ci95"]
nor_j = [o for o in nor if op_role_calls[o]["judge"]]
ret_j = [o for o in ret if op_role_calls[o]["judge"]]
out["no_retry_packages_with_no_judge_call"] = len(nor) - len(nor_j)
out["judge_loop_only"] = {"retry": summ(ret_j), "no_retry": summ(nor_j)}
out["judge_loop_only"]["ratio"] = round(out["judge_loop_only"]["retry"]["median_usd"] / out["judge_loop_only"]["no_retry"]["median_usd"], 2)
out["judge_loop_only"]["ratio_ci95"] = ratio_ci(ret_j, nor_j)
out["retry_role_usd_per_package"] = {"median": round(st.median(op_role_usd[o]["retry"] for o in ret), 4),
                                    "share_of_package_spend": prod.out["table_b_packet"]["roles"]["retry"]["share_of_spend"],
                                    "share_within_retry_packages": round(sum(op_role_usd[o]["retry"] for o in ret) / sum(op_usd[o] for o in ret), 3)}
json.dump(out, open(Path(__file__).resolve().parent / "out" / "retry_like_for_like.json", "w"), indent=1)
print(json.dumps(out, indent=1))
