from __future__ import annotations
import argparse,gc,json,pickle,random,sys
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import config
from data import EDGE_TYPES,StaticFeatureStore,load_snapshot,build_features
from model import Model
from negative_mining.curnm import CurNM,M

BASE_RUNS=ROOT/"runs"
OUT_RUNS=ROOT/"negative_mining"/"runs"
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
    et=("patent","sue","patent")
    val_pairs={pair(r) for r in val_records}
    old=ed.get(et,torch.empty((2,0),dtype=torch.long)).cpu()
    if old.numel()==0:
        return {"historical_val_mask_raw_removed":0,"historical_val_mask_unique_pairs":0}
    keep=[]; removed=set()
    for s,d in old.t().tolist():
        p=(int(s),int(d))
        if p in val_pairs: keep.append(False); removed.add(p)
        else: keep.append(True)
    keep=torch.tensor(keep,dtype=torch.bool)
    filtered=old[:,keep].contiguous(); ed[et]=filtered
    remaining={(int(s),int(d)) for s,d in filtered.t().tolist()}
    if remaining & val_pairs: raise RuntimeError("VAL leakage remains.")
    raw_removed=int((~keep).sum().item())
    print("[VAL MP MASK]",f"raw removed={raw_removed:,}",f"unique={len(removed):,}")
    return {"historical_val_mask_raw_removed":raw_removed,
            "historical_val_mask_unique_pairs":len(removed)}

def add_mp(ed,records):
    et=("patent","sue","patent")
    old=ed.get(et,torch.empty((2,0),dtype=torch.long)).cpu()
    old_unique=torch.unique(old.t(),dim=0).t().contiguous() if old.numel() else old.clone()
    pairs=sorted({pair(r) for r in records})
    add=torch.tensor(pairs,dtype=torch.long).t().contiguous() if pairs else torch.empty((2,0),dtype=torch.long)
    merged=torch.cat([old_unique,add],dim=1)
    if merged.numel(): merged=torch.unique(merged.t(),dim=0).t().contiguous()
    ed[et]=merged
    return {"historical_sue_raw":int(old.shape[1]),
            "historical_sue_unique":int(old_unique.shape[1]),
            "pseudo_mp_requested":len(pairs),
            "actual_new_pseudo_mp":int(merged.shape[1])-int(old_unique.shape[1]),
            "final_sue_unique":int(merged.shape[1])}

@torch.no_grad()
def fixed_val_metrics(model,z,records,device):
    ranks=[]
    for r in records:
        s,d=pair(r); neg=list(r.get("hard",[]))+list(r.get("random",[]))
        if not neg: raise ValueError("VAL hard/random 없음")
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
    for key in ["sup_all","val_fixed","mp_pseudo"]:
        positives|={pair(r) for r in round_data.get(key,[])}
    reserved=set(map(tuple,round_data.get("val_reserved_pairs",set())))
    pos_by_src={}; res_by_src={}
    for s,d in positives: pos_by_src.setdefault(int(s),set()).add(int(d))
    for s,d in reserved: res_by_src.setdefault(int(s),set()).add(int(d))
    return positives,reserved,pos_by_src,res_by_src

def build_pi_eval_pairs(
    train_pairs,
    pools,
    pos_by_src,
    res_by_src,
    n_pat,
    seed,
    n_struct=20,
):
    """
    Warm-start용 curriculum control set.

    각 positive source마다 기존 structural negative pool에서 최대
    n_struct개를 고정적으로 뽑는다. 부족한 경우에만 global random으로
    채운다. 한 번 만든 뒤 전체 epoch에서 동일하게 사용한다.

    주의:
    - 이것은 모델 학습용 negative가 아니다.
    - CurNM의 pi를 움직이기 위한 Train AP/AUC 측정용 고정 세트다.
    """
    rng=np.random.default_rng(seed)
    ps=[]; pd=[]; ns=[]; nd=[]

    for s,d in train_pairs:
        blocked={int(s)}
        blocked|=pos_by_src.get(int(s),set())
        blocked|=res_by_src.get(int(s),set())

        pool=pools.get(s,pools.get(str(s),[]))
        valid=np.asarray(
            [int(x) for x in pool if int(x) not in blocked],
            dtype=np.int64,
        )

        if len(valid)>n_struct:
            idx=rng.choice(
                len(valid),
                size=n_struct,
                replace=False,
            )
            chosen=valid[idx].tolist()
        else:
            chosen=valid.tolist()

        used=set(chosen)
        while len(chosen)<n_struct:
            x=int(rng.integers(0,n_pat))
            if x in blocked or x in used:
                continue
            chosen.append(x)
            used.add(x)

        ps.append(int(s))
        pd.append(int(d))
        ns.extend([int(s)]*len(chosen))
        nd.extend(chosen)

    return ps,pd,ns,nd


@torch.no_grad()
def train_ap_auc(
    model,
    fd,
    gd,
    pi_eval_pairs,
    device,
):
    """
    pi 제어용 Train AP/AUC.

    매 epoch 새 global random 1개를 뽑지 않고,
    시작 시 한 번 만든 고정 structural-negative 평가 세트를 사용한다.
    """
    model.eval()
    z=model.encode(fd,gd)

    ps,pd,ns,nd=pi_eval_pairs

    ps=torch.tensor(
        ps,dtype=torch.long,device=device)
    pd=torch.tensor(
        pd,dtype=torch.long,device=device)
    ns=torch.tensor(
        ns,dtype=torch.long,device=device)
    nd=torch.tensor(
        nd,dtype=torch.long,device=device)

    pos=model.score(z,ps,pd).sigmoid()
    neg=model.score(z,ns,nd).sigmoid()

    y_score=torch.cat(
        [pos,neg]).detach().cpu().numpy()
    y_true=np.concatenate([
        np.ones(len(pos)),
        np.zeros(len(neg)),
    ])

    ap=float(
        average_precision_score(y_true,y_score))
    auc=float(
        roc_auc_score(y_true,y_score))
    return ap,auc

def run(cutoff,round_idx,warm):
    seed_all(config.SEED+cutoff+round_idx*10000)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_dir=BASE_RUNS/f"cutoff_{cutoff}"/f"round_{round_idx:02d}"
    rpath=src_dir/"round_data.pkl"
    if not rpath.exists(): raise FileNotFoundError(rpath)
    rd=load_pkl(rpath)

    snap=load_snapshot(config.SNAP_DIR,cutoff)
    static=StaticFeatureStore.load(config.DATA_GNN)
    feats,info=build_features(snap,static); ed=edge_dict(snap)
    mask_stats=mask_fixed_val_from_historical_sue(ed,rd["val_fixed"])
    mp_stats=add_mp(ed,rd.get("mp_pseudo",[])); mp_stats.update(mask_stats)

    counts={k:int(v.shape[0]) for k,v in feats.items()}
    fd={k:v.to(device) for k,v in feats.items()}
    gd={"edge_dict":{k:v.to(device) for k,v in ed.items()},"counts":counts}

    c3=load_pkl(C3_DIR/f"cutoff_{cutoff}"/"c3_data.pkl")
    tr=rd["sup_all"]; va=rd["val_fixed"]; pools=rd["negative_pools"]
    n_pat=int(c3["num_patents"])
    _,_,pos_by_src,res_by_src=build_blocked(rd,c3)
    train_pairs=[pair(r) for r in tr]

    if len(train_pairs)!=len(set(train_pairs)): raise RuntimeError("SUP ALL duplicate")
    if set(train_pairs)&{pair(r) for r in va}: raise RuntimeError("SUP/VAL overlap")

    model=make_model(feats).to(device)
    if warm is None: raise ValueError("CurNM fine-tuning은 --warm-start 필요")
    state=torch.load(warm,map_location="cpu",weights_only=True)
    model.load_state_dict(state); print("warm start <-",warm)

    opt=torch.optim.AdamW(model.parameters(),lr=config.LR,weight_decay=config.WEIGHT_DECAY)
    rng=np.random.default_rng(config.SEED+900000+cutoff+round_idx*1000)
    miner=CurNM(pools,pos_by_src,res_by_src,n_pat,rng)

    pi_eval_pairs=build_pi_eval_pairs(
        train_pairs=train_pairs,
        pools=pools,
        pos_by_src=pos_by_src,
        res_by_src=res_by_src,
        n_pat=n_pat,
        seed=config.SEED+700000+cutoff+round_idx*1000,
        n_struct=20,
    )
    print(
        "[PI EVAL]",
        f"positive={len(pi_eval_pairs[0]):,}",
        f"fixed structural negatives={len(pi_eval_pairs[2]):,}",
        "per_positive=20",
    )

    ps=torch.tensor([s for s,_ in train_pairs],dtype=torch.long,device=device)
    pd=torch.tensor([d for _,d in train_pairs],dtype=torch.long,device=device)

    out_dir=OUT_RUNS/f"cutoff_{cutoff}"/f"round_{round_idx:02d}"
    out_dir.mkdir(parents=True,exist_ok=True)

    model.eval()
    with torch.no_grad():
        z0=model.encode(fd,gd); vm0=fixed_val_metrics(model,z0,va,device)

    best=vm0["MRR"]; best_epoch=0
    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    patience=0; history=[]

    print("="*90)
    print(f"CurNM | cutoff={cutoff} R{round_idx}")
    print(f"SUP={len(tr):,} VAL={len(va):,}")
    print(f"pseudo MP={len(rd.get('mp_pseudo',[])):,}")
    print(f"pseudo SUP={len(rd.get('sup_pseudo',[])):,}")
    print(f"M={M} | initial VAL MRR={best:.6f}")
    print("="*90)

    for epoch in range(1,config.EPOCHS+1):
        model.train(); z=model.encode(fd,gd)
        cand_src,cand_dst=miner.build_candidate_pairs(train_pairs)
        cs=torch.tensor(cand_src,dtype=torch.long,device=device)
        cd=torch.tensor(cand_dst,dtype=torch.long,device=device)

        pos_logits=model.score(z,ps,pd)
        cand_logits=model.score(z,cs,cd)
        sel_idx,n_sel,rand_prop=miner.select(cand_logits)
        sel_logits=cand_logits[sel_idx]

        rand_src,rand_dst=miner.sample_external_random(train_pairs)
        rs=torch.tensor(rand_src,dtype=torch.long,device=device)
        rd_t=torch.tensor(rand_dst,dtype=torch.long,device=device)
        rand_logits=model.score(z,rs,rd_t)

        loss_pos=F.binary_cross_entropy_with_logits(pos_logits,torch.ones_like(pos_logits))
        loss_sel=F.binary_cross_entropy_with_logits(sel_logits,torch.zeros_like(sel_logits))
        loss_rand=F.binary_cross_entropy_with_logits(rand_logits,torch.zeros_like(rand_logits))
        loss=loss_pos+(1.0-rand_prop)*loss_sel+rand_prop*loss_rand

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),config.GRAD_CLIP)
        opt.step()

        miner.update_cache(epoch,cand_src,cand_dst,cand_logits)
        train_ap,train_auc=train_ap_auc(
            model,
            fd,
            gd,
            pi_eval_pairs,
            device,
        )
        old_pi,new_pi=miner.policy.record(train_ap)
        miner.maybe_start_cache(epoch)

        row={"epoch":epoch,"loss":float(loss.item()),"train_AP":train_ap,
             "train_AUC":train_auc,"pi_before":old_pi,"pi_after":new_pi,
             "num_candidates":len(cand_dst),"num_selected":n_sel,
             "num_external_random":len(rand_dst),"rand_prop":rand_prop,
             "cache_started":miner.cache_started,"cache_sources":len(miner.cache)}

        if epoch%config.VAL_EVERY==0:
            model.eval()
            with torch.no_grad():
                zv=model.encode(fd,gd); vm=fixed_val_metrics(model,zv,va,device)

            row.update({"val_MRR":vm["MRR"],"MedianRank":vm["MedianRank"],
                        "Hits@10":vm["Hits@10"],"Hits@100":vm["Hits@100"]})
            mark=""

            if vm["MRR"]>best:
                best=vm["MRR"]; best_epoch=epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                patience=0; mark=" *"
            else:
                patience+=1

            print(f"epoch={epoch:04d} loss={loss.item():.4f} AP={train_ap:.4f} "
                  f"AUC={train_auc:.4f} pi={new_pi:.2f} sel={n_sel:,} "
                  f"rand={len(rand_dst):,} rp={rand_prop:.2f} "
                  f"cache={len(miner.cache):,} MRR={vm['MRR']:.6f}{mark}")

            if patience>=config.PATIENCE:
                print("early stopping"); history.append(row); break
        else:
            print(f"epoch={epoch:04d} loss={loss.item():.4f} AP={train_ap:.4f} "
                  f"AUC={train_auc:.4f} pi={new_pi:.2f} sel={n_sel:,} "
                  f"rand={len(rand_dst):,} rp={rand_prop:.2f}")

        history.append(row)

    ck=out_dir/"checkpoint.pt"; torch.save(best_state,ck)
    metrics={"cutoff":cutoff,"round":round_idx,"warm_start":str(warm),
             "best_epoch":best_epoch,"best_val_MRR":best,
             "initial_val_MRR":vm0["MRR"],"sup_pairs":len(tr),"val_pairs":len(va),
             "pseudo_mp_pairs":len(rd.get("mp_pseudo",[])),
             "pseudo_sup_pairs":len(rd.get("sup_pseudo",[])),
             "curnm_M":M,"pi_eval_neg_per_positive":20,
             "pi_eval_mode":"fixed_structural",
             "final_pi":miner.policy.pi,
             "cache_start_epoch":miner.cache_start_epoch,**mp_stats,**info}

    (out_dir/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    (out_dir/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
    print(json.dumps(metrics,indent=2)); print("saved:",ck)

    del model,opt,fd,gd; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return str(ck)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--cutoff",type=int,required=True)
    ap.add_argument("--round",type=int,required=True)
    ap.add_argument("--warm-start",required=True)
    a=ap.parse_args()
    run(a.cutoff,a.round,a.warm_start)
