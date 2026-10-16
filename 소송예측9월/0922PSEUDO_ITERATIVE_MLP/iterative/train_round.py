from __future__ import annotations
import argparse,gc,json,pickle,random,sys
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
    old_unique=(torch.unique(old.t(),dim=0).t().contiguous()
                if old.numel() else old.clone())

    pairs=sorted({pair(r) for r in records})
    add=(torch.tensor(pairs,dtype=torch.long).t().contiguous()
         if pairs else torch.empty((2,0),dtype=torch.long))

    merged=torch.cat([old_unique,add],dim=1)
    if merged.numel(): merged=torch.unique(merged.t(),dim=0).t().contiguous()
    ed[et]=merged

    historical_raw=int(old.shape[1])
    historical_unique=int(old_unique.shape[1])
    pseudo_requested=len(pairs)
    final_unique=int(merged.shape[1])

    return {
        "historical_sue_raw":historical_raw,
        "historical_sue_unique":historical_unique,
        "pseudo_mp_requested":pseudo_requested,
        "actual_new_pseudo_mp":final_unique-historical_unique,
        "final_sue_unique":final_unique,
    }

@torch.no_grad()
def fixed_val_metrics(model,z,records,device):
    ranks=[]
    for r in records:
        s,d=pair(r); neg=list(r.get("hard",[]))+list(r.get("random",[]))
        if not neg: raise ValueError("VAL record에 hard/random 후보가 없습니다.")
        src=torch.full((1+len(neg),),s,dtype=torch.long,device=device)
        dst=torch.tensor([d]+[int(x) for x in neg],dtype=torch.long,device=device)
        sc=model.score(z,src,dst); ranks.append(1+int((sc[1:]>sc[0]).sum().item()))
    a=np.asarray(ranks,dtype=np.int64)
    out={"MRR":float(np.mean(1/a)),"MedianRank":float(np.median(a))}
    for k in [1,10,50,100]: out[f"Hits@{k}"]=float(np.mean(a<=k))
    return out

def build_blocked(round_data,c3):
    positives=set(map(tuple,c3.get("all_positive_pairs",set())))
    for key in ["sup_all","val_fixed","mp_pseudo"]:
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
    # pool이 짧으면 안전한 global candidate로 hard quota 보충
    while len(hard)<config.N_HARD_TRAIN:
        x=int(rng.integers(0,n_pat))
        if x not in blocked and x not in hard: hard.append(x)
    blocked2=blocked|set(hard); rand=[]
    while len(rand)<config.N_RANDOM_TRAIN:
        x=int(rng.integers(0,n_pat))
        if x not in blocked2: rand.append(x); blocked2.add(x)
    return hard+rand

def run(cutoff,round_idx,warm=None):
    seed_all(config.SEED+cutoff+round_idx*10000)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rdir=RUNS_DIR/f"cutoff_{cutoff}"/f"round_{round_idx:02d}"
    rpath=rdir/"round_data.pkl"
    if not rpath.exists(): raise FileNotFoundError(f"round_data 없음: {rpath}")
    rd=load_pkl(rpath)
    if int(rd["cutoff"])!=cutoff or int(rd["round"])!=round_idx:
        raise ValueError("round_data의 cutoff/round가 CLI와 다릅니다.")

    snap=load_snapshot(config.SNAP_DIR,cutoff)
    static=StaticFeatureStore.load(config.DATA_GNN)
    feats,info=build_features(snap,static); ed=edge_dict(snap)

    # Fixed VAL must not be visible through historical sue MP.
    val_mp_mask_stats=mask_fixed_val_from_historical_sue(
        ed,
        rd["val_fixed"],
    )

    # Add allowed pseudo MP only after historical VAL masking.
    mp_stats=add_mp(ed,rd.get("mp_pseudo",[]))
    mp_stats.update(val_mp_mask_stats)
    counts={k:int(v.shape[0]) for k,v in feats.items()}
    fd={k:v.to(device) for k,v in feats.items()}
    gd={"edge_dict":{k:v.to(device) for k,v in ed.items()},"counts":counts}

    c3=load_pkl(C3_DIR/f"cutoff_{cutoff}"/"c3_data.pkl")
    tr=rd["sup_all"]; va=rd["val_fixed"]; pools=rd["negative_pools"]
    n_pat=int(c3["num_patents"])
    positives,reserved,pos_by_src,res_by_src=build_blocked(rd,c3)
    train_pairs=[pair(r) for r in tr]
    if len(train_pairs)!=len(set(train_pairs)): raise RuntimeError("SUP ALL에 중복 pair가 있습니다.")
    if set(train_pairs)&{pair(r) for r in va}: raise RuntimeError("SUP/VAL overlap 발견")

    model=make_model(feats).to(device)
    if warm:
        model.load_state_dict(torch.load(warm,map_location="cpu",weights_only=True)); print("warm start <-",warm)
    else: print("fresh init (no warm start)")
    opt=torch.optim.AdamW(model.parameters(),lr=config.LR,weight_decay=config.WEIGHT_DECAY)
    rng=np.random.default_rng(config.SEED+100000+cutoff+round_idx*1000)
    ps=torch.tensor([s for s,_ in train_pairs],dtype=torch.long,device=device)
    pd=torch.tensor([d for _,d in train_pairs],dtype=torch.long,device=device)

    print("="*100)
    print(f"TRAIN ROUND | cutoff={cutoff} round={round_idx}")
    print(f"SUP={len(tr):,} | VAL={len(va):,} | pseudo MP={len(rd.get('mp_pseudo',[])):,}")
    print(f"historical sue = {mp_stats['historical_sue_raw']:,} raw / {mp_stats['historical_sue_unique']:,} unique")
    print(f"pseudo MP      = {mp_stats['pseudo_mp_requested']:,} requested / {mp_stats['actual_new_pseudo_mp']:,} actually new")
    print(f"final sue      = {mp_stats['final_sue_unique']:,} unique")
    print(f"NEG / positive = {config.N_HARD_TRAIN} hard + {config.N_RANDOM_TRAIN} random")
    print(f"max epochs={config.EPOCHS} | val every={config.VAL_EVERY} | patience={config.PATIENCE}")
    print("="*100)

    best=-1.; best_state=None; best_epoch=0; patience=0
    for epoch in range(1,config.EPOCHS+1):
        model.train(); z=model.encode(fd,gd); ns=[]; nd=[]
        for s,d in train_pairs:
            hp=pools.get(s,pools.get(str(s),[]))
            neg=sample_negs(s,hp,pos_by_src,res_by_src,n_pat,rng)
            ns.extend([s]*len(neg)); nd.extend(neg)
        ns=torch.tensor(ns,dtype=torch.long,device=device); nd=torch.tensor(nd,dtype=torch.long,device=device)
        pos=model.score(z,ps,pd); neg=model.score(z,ns,nd)
        logits=torch.cat([pos,neg]); labels=torch.cat([torch.ones_like(pos),torch.zeros_like(neg)])
        loss=F.binary_cross_entropy_with_logits(logits,labels)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),config.GRAD_CLIP); opt.step()

        if epoch%config.VAL_EVERY==0:
            model.eval()
            with torch.no_grad(): zv=model.encode(fd,gd); vm=fixed_val_metrics(model,zv,va,device)
            mark=""
            if vm["MRR"]>best:
                best=vm["MRR"]; best_epoch=epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                patience=0; mark=" *"
            else: patience+=1
            print(f"epoch={epoch:04d} loss={loss.item():.4f} val_MRR={vm['MRR']:.6f} MedianRank={vm['MedianRank']:.1f} H@10={vm['Hits@10']:.4f} H@100={vm['Hits@100']:.4f}{mark} pat={patience}/{config.PATIENCE}")
            if patience>=config.PATIENCE: break

    if best_state is None: raise RuntimeError("no checkpoint")
    ck=rdir/"checkpoint.pt"; torch.save(best_state,ck)
    metrics={"cutoff":cutoff,"round":round_idx,"warm_start":str(warm) if warm else None,
             "best_epoch":best_epoch,"best_val_MRR":best,"sup_pairs":len(tr),"val_pairs":len(va),
             "pseudo_mp_pairs":len(rd.get("mp_pseudo",[])),"pseudo_sup_pairs":len(rd.get("sup_pseudo",[])),
             **mp_stats,
             "n_hard_train":int(config.N_HARD_TRAIN),"n_random_train":int(config.N_RANDOM_TRAIN),**info}
    (rdir/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2)); print("saved:",ck)
    del model,opt,fd,gd,z; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return str(ck)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--warm-start", default=None)

    a = ap.parse_args()

    warm = a.warm_start

    # Round 1부터는 바로 이전 Round의 checkpoint를 자동 사용
    if a.round > 0 and warm is None:
        prev_ckpt = (
            RUNS_DIR
            / f"cutoff_{a.cutoff}"
            / f"round_{a.round - 1:02d}"
            / "checkpoint.pt"
        )

        if not prev_ckpt.exists():
            raise FileNotFoundError(
                f"이전 round checkpoint가 없습니다:\n{prev_ckpt}"
            )

        warm = str(prev_ckpt)

    run(a.cutoff, a.round, warm)


