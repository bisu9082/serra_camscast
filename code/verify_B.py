"""Independent verification of Reviewer B's three most consequential counter-claims."""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from loso_v2 import PM10C, DATA, hav, FULL, NOXY

CAMS=["cams"]; HOUR=["cams","hsin","hcos"]; DOY=["cams","dsin","dcos"]

def run(R, cols, seed=0, early=False, time_block=False, nblk=6, guard_days=3):
    out={}
    for tgt in PM10C:
        src=[j for j in PM10C if j!=tgt and hav(PM10C[tgt],PM10C[j])>R]
        if len(src)<4: continue
        dft=DATA[tgt][0]
        if not time_block:
            Xtr=pd.concat([DATA[j][3][cols] for j in src])
            ytr=np.concatenate([DATA[j][0]["resid"].values for j in src])
            m=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_depth=6,
                                            random_state=seed,early_stopping=early).fit(Xtr,ytr)
            pred=m.predict(DATA[tgt][3][cols])
        else:
            # leave-location-and-time-out: predict each time block from other blocks only
            t0,t1=dft.index.min(),dft.index.max()
            edges=pd.date_range(t0,t1,periods=nblk+1)
            pred=np.full(len(dft),np.nan)
            for b in range(nblk):
                lo,hi=edges[b],edges[b+1]
                te=(dft.index>=lo)&(dft.index<(hi if b<nblk-1 else hi+pd.Timedelta("1h")))
                if te.sum()==0: continue
                g=pd.Timedelta(days=guard_days)
                Xtr=[];ytr=[]
                for j in src:
                    dj=DATA[j][0]; keep=(dj.index<lo-g)|(dj.index>hi+g)
                    if keep.sum()==0: continue
                    Xtr.append(DATA[j][3][cols][keep]); ytr.append(dj["resid"].values[keep])
                if not Xtr: continue
                m=HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05,max_depth=6,
                                                random_state=seed,early_stopping=early).fit(pd.concat(Xtr),np.concatenate(ytr))
                pred[te]=m.predict(DATA[tgt][3][cols][te])
        obs=dft["obs"].values; cams=dft["cams"].values
        ok=np.isfinite(pred)
        out[tgt]=(float(np.mean(np.abs(cams[ok]-obs[ok]))), float(np.mean(np.abs(cams[ok]+pred[ok]-obs[ok]))))
    return out

def summ(res):
    tg=sorted(res); raw=np.array([res[t][0] for t in tg]); tr=np.array([res[t][1] for t in tg])
    n=np.array([len(DATA[t][0]) for t in tg])
    g=100*(raw-tr)/raw; d=raw-tr
    return dict(pooled=100*(np.average(raw,weights=n)-np.average(tr,weights=n))/np.average(raw,weights=n),
                median=float(np.median(g)), wins=int((g>0).sum()), n=len(tg),
                dz=float(d.mean()/d.std(ddof=1)), tr=tr)

print("### CLAIM 1 — coordinate-free gain comes from day-of-year, not from a transferable mapping")
for nm,cols in [("cams only",CAMS),("cams+hour",HOUR),("cams+doy",DOY),("coordfree",NOXY),("full",FULL)]:
    for R in (0,100):
        s=summ(run(R,cols))
        print(f"  R={R:>3} {nm:<10} pooled {s['pooled']:+7.2f}%  median {s['median']:+7.2f}%  wins {s['wins']}/{s['n']}")
    print()

print("### CLAIM 2 — paired test, full vs coordinate-free (never reported in the paper)")
rng=np.random.default_rng(11)
for R in (0,25,50,100):
    a=summ(run(R,FULL)); b=summ(run(R,NOXY))
    d=a["tr"]-b["tr"]                      # positive = coordinate-free better
    idx=rng.integers(0,len(d),size=(2000,len(d)))
    ci=np.percentile(d[idx].mean(1),[2.5,97.5])
    w=stats.wilcoxon(a["tr"],b["tr"])
    print(f"  R={R:>3}  mean MAE diff {d.mean():+6.2f}  CI [{ci[0]:+.2f},{ci[1]:+.2f}]  "
          f"coordfree better at {int((d>0).sum())}/{len(d)}  Wilcoxon p={w.pvalue:.3f}")

print("\n### CLAIM 3 — spatiotemporal blocking (6 time blocks, 3-day guard band)")
for R in (0,100):
    for nm,cols in [("full",FULL),("coordfree",NOXY)]:
        s0=summ(run(R,cols)); s1=summ(run(R,cols,time_block=True))
        print(f"  R={R:>3} {nm:<10} space-only pooled {s0['pooled']:+7.2f}% (d_z {s0['dz']:+.2f}, {s0['wins']}/19)"
              f"   ->  +time-block {s1['pooled']:+7.2f}% (d_z {s1['dz']:+.2f}, {s1['wins']}/19)")
