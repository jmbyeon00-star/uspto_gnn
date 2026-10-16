from __future__ import annotations
import argparse,csv,gc,json,pickle,random,shutil,sys
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import config
from data import EDGE_TYPES,StaticFeatureStore,load_snapshot,build_features
from model import Model

RUNS_DIR=ROOT/"runs"
C3_DIR=ROOT/"fixed_data"/"c3_pools"

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def load_pkl(p):
    with open(p,"rb") as f: return pickle.load(f)

def edge_dict(snapshot):
    d=snapshot["data"]
    return {e:d[e].edge_index.cpu() for e in EDGE_TYPES if e in d.edge_types}

def make_model(feats):
    dims={k:int(v.shape[1]) for k,v in feats.items()}
    return Model(dims,config.HIDDEN,EDGE_TYPES,config.DROPOUT)

def pair(r): return int(r["src"]),int(r["dst"])
def patent_pair(r): return str(r["plt_patent"]),str(r["def_patent"])


def mask_fixed_val_from_historical_sue(ed,val_records):
    """
    Remove fixed-VAL positive pairs from historical
    ('patent','sue','patent') message-passing edges.
    Applied before pseudo MP is added.
    """
    et=("patent","sue","patent")
    val_pairs={pair(r) for r in val_records}
    old=ed.get(et,torch.empty((2,0),dtype=torch.long)).cpu()

    if old.numel()==0:
        print("[LEAKAGE CHECK] historical sue MP ∩ fixed VAL = 0")
        return {
            "historical_val_mask_raw_removed":0,
            "historical_val_mask_unique_pairs":0,
        }

    keep=[]
    removed_pairs=set()
    for s,d in old.t().tolist():
        p=(int(s),int(d))
        if p in val_pairs:
            keep.append(False)
            removed_pairs.add(p)
        else:
            keep.append(True)

    keep_t=torch.tensor(keep,dtype=torch.bool)
    filtered=old[:,keep_t].contiguous()
    ed[et]=filtered

    remaining={(int(s),int(d)) for s,d in filtered.t().tolist()}
    overlap_after=remaining & val_pairs
    if overlap_after:
        raise RuntimeError(
            "[LEAKAGE] historical sue MP ∩ fixed VAL != 0 "
            f"after masking: {sorted(overlap_after)[:20]}"
        )

    raw_removed=int((~keep_t).sum().item())
    print(
        f"[VAL MP MASK] historical sue raw removed={raw_removed:,} | "
        f"unique VAL pairs removed={len(removed_pairs):,}"
    )
    if removed_pairs:
        print(f"[VAL MP MASK] pairs={sorted(removed_pairs)[:20]}")
    print("[LEAKAGE CHECK] historical sue MP ∩ fixed VAL = 0")

    return {
        "historical_val_mask_raw_removed":raw_removed,
        "historical_val_mask_unique_pairs":len(removed_pairs),
    }

def add_mp(ed,records):
    et=("patent","sue","patent")
    old=ed.get(et,torch.empty((2,0),dtype=torch.long)).cpu()
    old_unique=(torch.unique(old.t(),dim=0).t().contiguous() if old.numel() else old.clone())
    pairs=sorted({pair(r) for r in records})
    add=(torch.tensor(pairs,dtype=torch.long).t().contiguous()
         if pairs else torch.empty((2,0),dtype=torch.long))
    merged=torch.cat([old_unique,add],dim=1)
    if merged.numel(): merged=torch.unique(merged.t(),dim=0).t().contiguous()
    ed[et]=merged
    return {
        "historical_sue_raw":int(old.shape[1]),
        "historical_sue_unique":int(old_unique.shape[1]),
        "pseudo_mp_requested":len(pairs),
        "actual_new_pseudo_mp":int(merged.shape[1])-int(old_unique.shape[1]),
        "final_sue_unique":int(merged.shape[1]),
    }

@torch.no_grad()
def fixed_val_metrics(model,z,records,device):
    ranks=[]
    for r in records:
        s,d=pair(r); neg=list(r.get("hard",[]))+list(r.get("random",[]))
        if not neg: raise ValueError("VAL record에 hard/random 후보가 없습니다.")
        src=torch.full((1+len(neg),),s,dtype=torch.long,device=device)
        dst=torch.tensor([d]+[int(x) for x in neg],dtype=torch.long,device=device)
        sc=model.score(z,src,dst)
        ranks.append(1+int((sc[1:]>sc[0]).sum().item()))
    a=np.asarray(ranks,dtype=np.int64)
    out={"MRR":float(np.mean(1/a)),"MedianRank":float(np.median(a))}
    for k in [1,10,50,100]: out[f"Hits@{k}"]=float(np.mean(a<=k))
    return out

def build_blocked(round_data,c3):
    positives=set(map(tuple,c3.get("all_positive_pairs",set())))
    for key in ["sup_all","val_fixed","mp_inherited","mp_pseudo"]:
        positives|={pair(r) for r in round_data.get(key,[])}
    reserved=set(map(tuple,round_data.get("val_reserved_pairs",set())))
    pos_by_src={}; res_by_src={}
    for s,d in positives: pos_by_src.setdefault(int(s),set()).add(int(d))
    for s,d in reserved: res_by_src.setdefault(int(s),set()).add(int(d))
    return positives,reserved,pos_by_src,res_by_src

def sample_negs(s,hard_pool,pos_by_src,res_by_src,n_pat,rng):
    blocked={s}|pos_by_src.get(s,set())|res_by_src.get(s,set())
    pool=[int(x) for x in hard_pool if int(x) not in blocked]
    rng.shuffle(pool); hard=pool[:config.N_HARD_TRAIN]
    while len(hard)<config.N_HARD_TRAIN:
        x=int(rng.integers(0,n_pat))
        if x not in blocked and x not in hard: hard.append(x)
    blocked2=blocked|set(hard); rand=[]
    while len(rand)<config.N_RANDOM_TRAIN:
        x=int(rng.integers(0,n_pat))
        if x not in blocked2: rand.append(x); blocked2.add(x)
    return hard+rand

def normalize_inherited_records(payload):
    records=payload.get("inherited_records",[])
    pairs=payload.get("inherited_pairs",[])
    patent_pairs=payload.get("inherited_patent_pairs",[])

    if patent_pairs:
        if len(patent_pairs)!=len(pairs):
            raise RuntimeError("inherited_patent_pairs/pairs 길이 불일치")
        idx_by_pat={tuple(map(str,pp)):tuple(map(int,p)) for pp,p in zip(patent_pairs,pairs)}
    else:
        if len(records)!=len(pairs):
            raise RuntimeError("inherited_records/pairs 길이 불일치")
        idx_by_pat={patent_pair(r):tuple(map(int,p)) for r,p in zip(records,pairs)}

    out=[]; seen=set()
    for r in records:
        pp=patent_pair(r)
        if pp in seen: continue
        seen.add(pp)
        if pp not in idx_by_pat:
            raise RuntimeError(f"target index mapping 없음: {pp}")
        x=dict(r); ts,td=idx_by_pat[pp]
        if "src" in x: x.setdefault("source_src",x["src"])
        if "dst" in x: x.setdefault("source_dst",x["dst"])
        x["src"],x["dst"]=ts,td
        out.append(x)
    return out


def build_candidate_round_data(cutoff,source_cutoff,source_round):
    base_path=RUNS_DIR/f"cutoff_{cutoff}"/"round_00"/"round_data.pkl"
    inherited_path=(RUNS_DIR/f"predecessor_{source_cutoff}_to_{cutoff}"
                    /f"source_round_{source_round:02d}"/"inherited.pkl")
    if not base_path.exists():
        raise FileNotFoundError(f"target R0 round_data 없음: {base_path}")
    if not inherited_path.exists():
        raise FileNotFoundError(f"inherited.pkl 없음: {inherited_path}")

    rd=load_pkl(base_path)
    inherited_payload=load_pkl(inherited_path)
    inherited=normalize_inherited_records(inherited_payload)

    # Predecessor selection에서는 target-fold 신규 pseudo를 절대 넣지 않는다.
    rd["mp_inherited"]=inherited
    rd["mp_pseudo"]=[]
    rd["sup_pseudo"]=[]
    rd["round"]=0

    expected=len(inherited_payload.get("inherited_pairs",[]))
    if len(inherited)!=expected:
        raise RuntimeError("inherited count mismatch")

    return rd,base_path,inherited_path,inherited_payload


def run_candidate(cutoff,source_cutoff,source_round):
    round_idx=0
    seed_all(config.SEED+cutoff+source_round*10000)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    warm=RUNS_DIR/f"cutoff_{source_cutoff}"/f"round_{source_round:02d}"/"checkpoint.pt"
    if not warm.exists():
        raise FileNotFoundError(f"warm-start checkpoint 없음: {warm}")

    out_root=RUNS_DIR/f"predecessor_{source_cutoff}_to_{cutoff}"
    rdir=out_root/f"source_round_{source_round:02d}"
    rdir.mkdir(parents=True,exist_ok=True)

    rd,base_path,inherited_path,inherited_payload=build_candidate_round_data(
        cutoff,source_cutoff,source_round
    )

    snap=load_snapshot(config.SNAP_DIR,cutoff)
    static=StaticFeatureStore.load(config.DATA_GNN)
    feats,info=build_features(snap,static); ed=edge_dict(snap)

    val_mp_mask_stats=mask_fixed_val_from_historical_sue(ed,rd["val_fixed"])

    # Target historical GT_MP + inherited pseudo only.
    inherited=rd.get("mp_inherited",[])
    mp_stats=add_mp(ed,inherited)
    mp_stats.update(val_mp_mask_stats)

    counts={k:int(v.shape[0]) for k,v in feats.items()}
    fd={k:v.to(device) for k,v in feats.items()}
    gd={"edge_dict":{k:v.to(device) for k,v in ed.items()},"counts":counts}

    c3=load_pkl(C3_DIR/f"cutoff_{cutoff}"/"c3_data.pkl")
    tr=rd["sup_all"]; va=rd["val_fixed"]; pools=rd["negative_pools"]
    n_pat=int(c3["num_patents"])
    _,_,pos_by_src,res_by_src=build_blocked(rd,c3)
    train_pairs=[pair(r) for r in tr]

    if len(train_pairs)!=len(set(train_pairs)):
        raise RuntimeError("SUP ALL에 중복 pair가 있습니다.")
    if set(train_pairs)&{pair(r) for r in va}:
        raise RuntimeError("SUP/VAL overlap 발견")

    pat2idx={str(k):int(v) for k,v in snap["pat2idx"].items()}
    idx2pat={int(v):str(k) for k,v in pat2idx.items()}
    idx_to_pat=lambda p:(idx2pat[int(p[0])],idx2pat[int(p[1])])

    inherited_pat={patent_pair(r) for r in inherited}
    gt_pat={idx_to_pat(p) for p in c3.get("all_positive_pairs",set())}
    val_pat={idx_to_pat(pair(r)) for r in va}
    if inherited_pat & gt_pat:
        raise RuntimeError(f"inherited ∩ GT 발견: {len(inherited_pat & gt_pat)}")
    if inherited_pat & val_pat:
        raise RuntimeError(f"inherited ∩ VAL 발견: {len(inherited_pat & val_pat)}")

    model=make_model(feats).to(device)
    model.load_state_dict(torch.load(warm,map_location="cpu",weights_only=True))
    opt=torch.optim.AdamW(model.parameters(),lr=config.LR,weight_decay=config.WEIGHT_DECAY)
    rng=np.random.default_rng(config.SEED+100000+cutoff+source_round*1000)
    ps=torch.tensor([s for s,_ in train_pairs],dtype=torch.long,device=device)
    pd=torch.tensor([d for _,d in train_pairs],dtype=torch.long,device=device)

    print("\n"+"="*100)
    print(f"PREDECESSOR | {source_cutoff} R{source_round} -> {cutoff}")
    print(f"WARM       <- {warm}")
    print(f"INHERITED  <- {inherited_path}")
    print(f"OUTPUT     -> {rdir}")
    print(f"SUP={len(tr):,} | VAL={len(va):,} | inherited MP={len(inherited):,}")
    print(f"historical sue = {mp_stats['historical_sue_raw']:,} raw / {mp_stats['historical_sue_unique']:,} unique")
    print(f"inherited MP  = {mp_stats['pseudo_mp_requested']:,} requested / {mp_stats['actual_new_pseudo_mp']:,} actually new")
    print(f"final sue     = {mp_stats['final_sue_unique']:,} unique")
    print(f"NEG / positive = {config.N_HARD_TRAIN} hard + {config.N_RANDOM_TRAIN} random")
    print(f"max epochs={config.EPOCHS} | val every={config.VAL_EVERY} | patience={config.PATIENCE}")
    print("="*100)

    best=-1.; best_state=None; best_epoch=0; best_metrics=None; patience=0
    for epoch in range(1,config.EPOCHS+1):
        model.train(); z=model.encode(fd,gd); ns=[]; nd=[]
        for s,d in train_pairs:
            hp=pools.get(s,pools.get(str(s),[]))
            neg=sample_negs(s,hp,pos_by_src,res_by_src,n_pat,rng)
            ns.extend([s]*len(neg)); nd.extend(neg)

        ns=torch.tensor(ns,dtype=torch.long,device=device)
        nd=torch.tensor(nd,dtype=torch.long,device=device)
        pos=model.score(z,ps,pd); neg=model.score(z,ns,nd)
        logits=torch.cat([pos,neg])
        labels=torch.cat([torch.ones_like(pos),torch.zeros_like(neg)])
        loss=F.binary_cross_entropy_with_logits(logits,labels)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),config.GRAD_CLIP)
        opt.step()

        if epoch%config.VAL_EVERY==0:
            model.eval()
            with torch.no_grad():
                zv=model.encode(fd,gd); vm=fixed_val_metrics(model,zv,va,device)
            mark=""
            if vm["MRR"]>best:
                best=vm["MRR"]; best_epoch=epoch; best_metrics=dict(vm)
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                patience=0; mark=" *"
            else:
                patience+=1
            print(f"epoch={epoch:04d} loss={loss.item():.4f} val_MRR={vm['MRR']:.6f} "
                  f"MedianRank={vm['MedianRank']:.1f} H@10={vm['Hits@10']:.4f} "
                  f"H@100={vm['Hits@100']:.4f}{mark} pat={patience}/{config.PATIENCE}")
            if patience>=config.PATIENCE: break

    if best_state is None:
        raise RuntimeError("no checkpoint")

    ck=rdir/"checkpoint.pt"
    torch.save(best_state,ck)

    stats=inherited_payload.get("stats",{})
    metrics={
        "cutoff":cutoff,
        "source_cutoff":source_cutoff,
        "source_round":source_round,
        "warm_start":str(warm),
        "base_round_data":str(base_path),
        "inherited_file":str(inherited_path),
        "checkpoint":str(ck),
        "source_pseudo":int(stats.get("source_pseudo",len(inherited))),
        "remapped":int(stats.get("remapped",len(inherited))),
        "missing":int(stats.get("missing",0)),
        "inherited_mp":len(inherited),
        "best_epoch":best_epoch,
        "best_val_MRR":best,
        "best_val_metrics":best_metrics,
        "sup_pairs":len(tr),
        "val_pairs":len(va),
        "target_new_pseudo_mp":0,
        "target_new_pseudo_sup":0,
        **mp_stats,
        "n_hard_train":int(config.N_HARD_TRAIN),
        "n_random_train":int(config.N_RANDOM_TRAIN),
        **info,
    }
    (rdir/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(f"\n[DONE] source R{source_round} | best_epoch={best_epoch} | best_val_MRR={best:.6f}")

    del model,opt,fd,gd,z
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return metrics


def save_comparison(rows,out_root):
    fields=[
        "source_round","source_pseudo","remapped","missing","inherited_mp",
        "historical_sue_unique","actual_new_pseudo_mp","final_sue_unique",
        "sup_pairs","val_pairs","best_epoch","best_val_MRR"
    ]
    with open(out_root/"comparison.csv","w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k:r.get(k) for k in fields})


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source-cutoff",type=int,required=True)
    ap.add_argument("--cutoff",type=int,required=True)
    ap.add_argument("--start-round",type=int,default=0)
    ap.add_argument("--end-round",type=int,default=10)
    a=ap.parse_args()

    if a.start_round<0 or a.end_round<a.start_round:
        raise ValueError("round 범위를 확인하세요.")

    out_root=RUNS_DIR/f"predecessor_{a.source_cutoff}_to_{a.cutoff}"
    out_root.mkdir(parents=True,exist_ok=True)

    rows=[]
    for source_round in range(a.start_round,a.end_round+1):
        rows.append(run_candidate(a.cutoff,a.source_cutoff,source_round))
        save_comparison(rows,out_root)

    best=max(rows,key=lambda x:x["best_val_MRR"])
    selected={
        "source_cutoff":a.source_cutoff,
        "target_cutoff":a.cutoff,
        "selected_source_round":best["source_round"],
        "source_checkpoint":best["warm_start"],
        "predecessor_checkpoint":best["checkpoint"],
        "inherited_file":best["inherited_file"],
        "inherited_mp":best["inherited_mp"],
        "best_epoch":best["best_epoch"],
        "best_val_MRR":best["best_val_MRR"],
        "best_val_metrics":best["best_val_metrics"],
    }
    with open(out_root/"selected.pkl","wb") as f:
        pickle.dump(selected,f)

    selected_ck=out_root/"selected_checkpoint.pt"
    shutil.copy2(best["checkpoint"],selected_ck)

    print("\n"+"="*100)
    print("PREDECESSOR COMPARISON")
    print("="*100)
    for r in rows:
        mark="  <-- SELECTED" if r["source_round"]==best["source_round"] else ""
        print(f"R{r['source_round']:02d} | inherited={r['inherited_mp']:3d} | "
              f"epoch={r['best_epoch']:4d} | VAL_MRR={r['best_val_MRR']:.6f}{mark}")
    print("="*100)
    print("comparison ->",out_root/"comparison.csv")
    print("selected   ->",out_root/"selected.pkl")
    print("checkpoint ->",selected_ck)


if __name__=="__main__":
    main()
