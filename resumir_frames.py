import argparse,csv,statistics as s
from pathlib import Path
from simulador import write_csv
p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();folder=Path(a.folder);groups={}
with (folder/'frames.csv').open(encoding='utf8') as f:
 for r in csv.DictReader(f): groups.setdefault((r['scenario'],r['mode'],r['seed']),[]).append(r)
def p95(v): return sorted(v)[int(.95*(len(v)-1))] if v else None
rows=[]
for (sc,m,seed),g in groups.items():
 iv=[float(r['interval_ms']) for r in g if r['interval_ms']];work=[float(r['work_ms']) for r in g];late=[float(r['late_ms']) for r in g];miss=sum(int(r['miss']) for r in g)
 rows.append(dict(scenario=sc,mode=m,seed=seed,n=len(g),hz=1000/s.mean(iv) if iv else None,interval_p95_ms=p95(iv),work_p95_ms=p95(work),late_p95_ms=p95(late),misses=miss,miss_fraction=miss/len(g),midphase_commits=sum(int(r['commit']) and not int(r['safe']) for r in g)))
write_csv(folder/'metricas_60hz.csv',rows)
print(len(rows),'episódios temporais; métricas por episódio (não tratar frames correlacionados como observações independentes).')
