from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PY=sys.executable

PATH=[(2004,1),(2006,7),(2008,7),(2010,2)]

def base_checkpoint(cutoff,r):
    return ROOT/"runs"/f"cutoff_{cutoff}"/f"round_{r:02d}"/"checkpoint.pt"

def curnm_checkpoint(cutoff,r):
    return ROOT/"negative_mining"/"runs"/f"cutoff_{cutoff}"/f"round_{r:02d}"/"checkpoint.pt"

prev=None

for i,(cutoff,r) in enumerate(PATH):
    warm=base_checkpoint(cutoff,r) if i==0 else prev
    if not warm.exists(): raise FileNotFoundError(warm)

    print("="*90)
    print(f"CurNM {cutoff} R{r}")
    print("warm <-",warm)
    print("="*90)

    cmd=[PY,str(ROOT/"negative_mining"/"train_curnm.py"),
         "--cutoff",str(cutoff),"--round",str(r),"--warm-start",str(warm)]

    subprocess.run(cmd,check=True,cwd=ROOT)
    prev=curnm_checkpoint(cutoff,r)

print("\nFINAL CHECKPOINT:")
print(prev)
