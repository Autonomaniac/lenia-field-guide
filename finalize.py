"""Merge a survivor record and its characterisation into specimens/<name>.json.
usage: python3 finalize.py survivors.jsonl characterized.jsonl cid:name [cid:name ...]"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
S = {str(json.loads(l)["cid"]): json.loads(l) for l in open(sys.argv[1])}
C = {str(json.loads(l)["cid"]): json.loads(l) for l in open(sys.argv[2])}


def lineage(cid):
    out = []
    f = S.get(cid)
    while f and f.get("parent"):
        out.append(str(f["parent"])); f = S.get(str(f["parent"]))
    return out


for arg in sys.argv[3:]:
    cid, name = arg.split(":")
    f, c = S[cid], C[cid]
    assert c["ok"], f"{cid} failed characterisation: {c.get('why')}"
    rec = dict(cid=cid, params=f["params"], pattern=f["pattern"], origin=f.get("origin"),
               lineage=lineage(cid), born_by_division=f.get("born_by_division"),
               search_desc=f["desc"], desc=c["desc"], mass_range=c["mass_range"], mirror=c["mirror"],
               rot_corr=c.get("rot_corr"), max_bodies=c["max_bodies"], frac_one_body=c["frac_one_body"],
               mass_trace=c["mass_trace"], encounters=c.get("encounters", []),
               graded=c.get("graded"), sensitivity=c.get("sensitivity"))
    json.dump(rec, open(os.path.join(HERE, "specimens", name + ".json"), "w"), indent=1)
    print("wrote", name)
