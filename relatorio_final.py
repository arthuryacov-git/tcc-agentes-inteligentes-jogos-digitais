import argparse,csv,json,subprocess,sys,math
from pathlib import Path
from scipy.stats import binom,binomtest
import numpy as np
from simulador import SCENARIOS,write_csv
from analisar import holm

def power(n,discordance=.35,diff=.15,alpha=.05/6):
    result=0.; p=(discordance+diff)/(2*discordance)
    for k in range(1,n+1):
        j=np.arange(k+1); reject=2*binom.cdf(np.minimum(j,k-j),k,.5)<=alpha
        result+=binom.pmf(k,n,discordance)*sum(binom.pmf(j[reject],k,p))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args(); folder=Path(a.folder); conf=json.loads((folder/'config.json').read_text()); n=conf['n']; models=[x['model'] for x in conf['models']]
    with (folder/'episodios.csv').open(encoding='utf8') as f: data=list(csv.DictReader(f))
    groups={}; expected=set(range(conf['seed'],conf['seed']+n))
    for r in data:
        key=(r['scenario'],r['mode']); g=groups.setdefault(key,{})
        if int(r['seed']) in g: raise ValueError('Semente duplicada')
        g[int(r['seed'])]=r
    required=[(s,m) for s in SCENARIOS for m in ['fsm_bt','planejador']+[m+':'+model for model in models for m in ['llm_puro','llm_hibrido']]]
    missing=[(s,m,len(groups.get((s,m),{}))) for s,m in required if set(groups.get((s,m),{}))!=expected]
    print('Grupos incompletos:',missing)
    if set(groups)!=set(required): raise RuntimeError('Grupos inesperados ou ausentes')
    if any(r.get('protocol')!='V5.0' for r in data): raise RuntimeError('Mistura de protocolos')
    if missing: raise RuntimeError('Amostra incompleta: retome a execução antes da análise confirmatória.')
    subprocess.run([sys.executable,str(Path(__file__).with_name('analisar.py')),str(folder)],check=True)
    if conf['realtime'] or n!=200 or conf['baseline_only'] or conf['seed']!=20261119 or conf['period']!=4 or conf.get('tick_s')!=1.0 or conf.get('warmups')!=2 or set(models)!={'qwen2.5:3b','llama3.2:3b'}:
        print('Análise descritiva/exploratória apenas: desenho não confirmatório.'); return
    rng=np.random.default_rng(431); tests=[]
    for s in SCENARIOS:
        for model in models:
            a=groups[s,'llm_puro:'+model]; b=groups[s,'llm_hibrido:'+model]; x=np.array([int(a[i]['success']) for i in sorted(expected)]); y=np.array([int(b[i]['success']) for i in sorted(expected)]); d=y-x; ab=int(sum(d==-1));ba=int(sum(d==1));lo,hi=np.quantile(d[rng.integers(0,n,(10000,n))].mean(axis=1),[.025,.975]);tests.append(dict(scenario=s,model=model,n=n,puro=float(x.mean()),hibrido=float(y.mean()),diferenca=float(d.mean()),ci_low=float(lo),ci_high=float(hi),puro_only=ab,hibrido_only=ba,p_mcnemar=float(binomtest(ba,ab+ba,.5).pvalue) if ab+ba else 1.))
    if tests: holm(tests,'p_mcnemar','p_holm');write_csv(folder/'comparacoes_primarias.csv',tests)
    elif not conf['baseline_only']: raise RuntimeError('Nenhum LLM')
    print('Episódios completos:',len(data),'LLMs:',models,'poder planejado:',round(power(n),4),'(6 comparações; parâmetros hipotéticos)')
    if not models: print('SOMENTE BASELINES; NÃO É EXPERIMENTO COM LLM.')
if __name__=='__main__': main()
