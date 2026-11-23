import argparse,csv,math,itertools,statistics as st
from pathlib import Path
import numpy as np
from scipy.stats import binomtest, wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from simulador import write_csv

def mean(v): return st.mean(v) if v else None
def sd(v): return st.stdev(v) if len(v)>1 else None
def vals(rows,k): return [float(r[k]) for r in rows if r[k]!='']
def wilson(k,n):
    z=1.95996398454; p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/d
    return c-h,c+h

def holm(rows,k,out):
    ix=sorted(range(len(rows)),key=lambda i:rows[i][k]); m=len(ix); last=0
    for j,i in enumerate(ix):
        last=max(last,min(1,(m-j)*rows[i][k])); rows[i][out]=last

def main():
    p=argparse.ArgumentParser(); p.add_argument('folder',nargs='?',default='resultados/piloto'); a=p.parse_args(); path=Path(a.folder)
    with (path/'episodios.csv').open(encoding='utf-8') as f: data=list(csv.DictReader(f))
    groups={}; rng=np.random.default_rng(431)
    for r in data: groups.setdefault((r['scenario'],r['mode']),[]).append(r)
    desc=[]; tests=[]
    for (sc,m),rows in sorted(groups.items()):
        n=len(rows); k=sum(int(x['success']) for x in rows); l,u=wilson(k,n)
        d=dict(scenario=sc,mode=m,n=n,success=k,rate=k/n,ci_low=l,ci_high=u,adapted_rate=mean(vals(rows,'adapted')),completed_turns_mean=mean(vals([r for r in rows if r['success']=='1'],'turns')),adaptation_turns_mean=mean(vals(rows,'adaptation_turns')),adaptation_n=len(vals(rows,'adaptation_turns')))
        for v in ['turns','decision_ms','body_ms','cpu_ms','calls','tokens_in','tokens_out','invalid_actions','persona_violations','reversals','stale_responses','waiting_ticks','wait_actions','voluntary_waits','body_blocked_waits','invalid_waits','inference_waits','longest_wait_streak','revisited_states','wait_no_change','body_overrides','body_goal_actions','llm_direct_actions','used_responses','actions_during_inference','scheduled_overlap_actions','context_rejected_turns','redundant_goal_turns','redundant_goal_responses','redundant_at_handoff','invalid_outputs','wall_ms']:
            x=vals(rows,v); d[v+'_mean']=mean(x); d[v+'_sd']=sd(x); d[v+'_median']=st.median(x) if x else None
        desc.append(d)
    for sc in sorted({s for s,m in groups}):
        for ma,mb in itertools.combinations(sorted(m for s,m in groups if s==sc),2):
            aa={r['seed']:r for r in groups[sc,ma]}; bb={r['seed']:r for r in groups[sc,mb]}; seeds=sorted(aa.keys()&bb.keys()); n=len(seeds)
            if not n: continue
            # paired differences B-A; bootstrap resamples seeds, not turns.
            x=np.array([int(aa[s]['success']) for s in seeds]); y=np.array([int(bb[s]['success']) for s in seeds]); delta=y-x; ab=int(sum((x==1)&(y==0))); ba=int(sum((x==0)&(y==1)))
            bs=delta[rng.integers(0,n,size=(5000,n))].mean(axis=1); lo,hi=np.quantile(bs,[.025,.975])
            # Secondary endpoint: fraction of time budget used, includes failed episodes; not speed among successes.
            t=np.array([float(bb[s]['turns'])/float(bb[s]['deadline'])-float(aa[s]['turns'])/float(aa[s]['deadline']) for s in seeds]); wp=float(wilcoxon(t).pvalue) if np.any(t!=0) else 1.0
            tests.append(dict(scenario=sc,A=ma,B=mb,n=n,A_only_success=ab,B_only_success=ba,rate_difference_B_minus_A=float(delta.mean()),bootstrap_ci_low=float(lo),bootstrap_ci_high=float(hi),p_mcnemar=float(binomtest(ba,ab+ba,.5).pvalue) if ab+ba else 1.,budget_fraction_difference=float(t.mean()),p_wilcoxon=wp))
    holm(tests,'p_mcnemar','p_mcnemar_holm'); holm(tests,'p_wilcoxon','p_wilcoxon_holm')
    write_csv(path/'estatistica_descritiva.csv',desc); write_csv(path/'comparacoes_exploratorias.csv',tests)
    scenarios=sorted({r['scenario'] for r in desc}); modes=sorted({r['mode'] for r in desc}); fig,ax=plt.subplots(figsize=(10,5)); w=.8/len(modes)
    for i,m in enumerate(modes):
        rs=[next((r for r in desc if r['scenario']==s and r['mode']==m),None) for s in scenarios]; xx=np.arange(len(scenarios))+(i-(len(modes)-1)/2)*w; yy=[r['rate']*100 if r else np.nan for r in rs]; ax.bar(xx,yy,w,label=m)
        for x,y,r in zip(xx,yy,rs):
            if r: ax.errorbar(x,y,yerr=[[y-r['ci_low']*100],[r['ci_high']*100-y]],fmt='none',ecolor='black',capsize=2)
    ax.set_xticks(range(len(scenarios)),scenarios); ax.set_ylim(0,108); ax.set_ylabel('Sucesso (%) — IC95% Wilson'); ax.legend(fontsize=8); ax.set_title('Resultados por cenário — V5'); fig.tight_layout(); fig.savefig(path/'sucesso.png',dpi=160); plt.close(fig)
    print(path, 'grupos=',len(desc),'comparações=',len(tests))
if __name__=='__main__': main()
