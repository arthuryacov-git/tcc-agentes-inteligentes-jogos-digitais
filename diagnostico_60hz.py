import argparse,hashlib,json,os,time
from pathlib import Path
from experimento import Ollama,run_realtime,run_episode,PROTOCOL,export,load_records
from simulador import SCENARIOS,write_csv

MODELS=('qwen2.5:3b','llama3.2:3b')

def diagnostic(client,seed):
    # Controlled initial goal: isolates overlap from the quality of the initial LLM choice.
    # A fresh real LLM response may replace it at a safe boundary.
    r=run_realtime('B_obstaculos',seed,'llm_hibrido',client,tick_s=.05,period=2,frame_hz=60,bootstrap_goal='COLLECT')
    for row in r['rows']:
        at=row['action_time_ms']; row['actual_inference_overlap']=int(any(c['worker_start_ms']<=at<c['worker_finish_ms'] for c in r['calls']))
    r['summary'].update(protocol=PROTOCOL,diagnostic='controlled_initial_goal',backend='ollama',initial_goal='COLLECT',actions_during_inference=sum(x['actual_inference_overlap'] and x['executed']!='WAIT' for x in r['rows']))
    return r

def fingerprint():
    return {n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in ('experimento.py','simulador.py','diagnostico_60hz.py')}

def gate(folder,config):
    folder=Path(folder); cfg=json.loads((folder/'config.json').read_text(encoding='utf8')); rs=load_records(folder/'registros.jsonl')
    if cfg['protocol']!=PROTOCOL or cfg['models']!=config['models'] or cfg['sha256']!=fingerprint(): raise RuntimeError('Diagnóstico temporal incompatível com o protocolo/modelos atuais.')
    expected={(m,s,seed,kind) for m in MODELS for s,seed,kind in [(s,20261005,'natural') for s in SCENARIOS]+[('B_obstaculos',20261006,'controlled_initial_goal'),('B_obstaculos',20261007,'controlled_initial_goal')]}
    actual=set(); totals={m:0 for m in MODELS}
    for r in rs:
        s=r['summary']; m=s['model']; key=(m,s['scenario'],s['seed'],s['diagnostic']);
        if key in actual: raise RuntimeError('Diagnóstico duplicado')
        actual.add(key); calls=r['calls']; frames=r['frames']; digests={x['model']:x['digest'] for x in config['models']}
        if s['protocol']!=PROTOCOL or s['digest']!=digests.get(m) or not calls or not frames or any(f['commit'] and not f['safe'] for f in frames): raise RuntimeError('Diagnóstico inconsistente')
        if any(c.get('model')!=m or c.get('digest')!=s['digest'] or c.get('tokens_in') is None or c.get('tokens_out') is None for c in calls): raise RuntimeError('Identidade ou metadados da inferência ausentes')
        overlap=sum(x['executed']!='WAIT' and any(c['worker_start_ms']<=x['action_time_ms']<c['worker_finish_ms'] for c in calls) for x in r['rows'])
        if overlap!=s['actions_during_inference']: raise RuntimeError('Sobreposição divergente')
        if s['diagnostic']=='controlled_initial_goal': totals[m]+=overlap
    if actual!=expected: raise RuntimeError('Diagnóstico temporal incompleto')
    print('Ações úteis durante inferência real, diagnóstico controlado:',totals)
    if not all(totals.values()): raise RuntimeError('Sobreposição não demonstrada para um dos modelos. Envie os registros para revisão; não é exigido sucesso da missão.')
    return totals

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='resultados/v5_verificacao_60hz');p.add_argument('--config',default='resultados/v5_verificacao/config.json');p.add_argument('--gate',action='store_true');a=p.parse_args();folder=Path(a.out);base=json.loads(Path(a.config).read_text(encoding='utf8'))
    if a.gate: gate(folder,base);return
    folder.mkdir(parents=True,exist_ok=True);log=folder/'registros.jsonl';cf=folder/'config.json';rs=load_records(log); clients=[Ollama(base['url'],m) for m in MODELS]; meta=[c.preflight() for c in clients]
    if meta!=base['models']: raise RuntimeError('Modelos diferentes da verificação lógica')
    cfg=dict(protocol=PROTOCOL,models=meta,sha256=fingerprint(),natural_tick_s=1.0,controlled_tick_s=.05,frame_hz=60,description='Diagnóstico técnico, fora da amostra oficial')
    if cf.exists() and json.loads(cf.read_text(encoding='utf8'))!=cfg: raise RuntimeError('Configuração mudou; preservar dados anteriores')
    if not cf.exists(): cf.write_text(json.dumps(cfg,indent=2),encoding='utf8')
    done={(r['summary']['model'],r['summary']['scenario'],r['summary']['seed'],r['summary']['diagnostic']) for r in rs}
    for c in clients:
        for sc,seed,kind in [(s,20261005,'natural') for s in SCENARIOS]+[('B_obstaculos',20261006,'controlled_initial_goal'),('B_obstaculos',20261007,'controlled_initial_goal')]:
            if (c.model,sc,seed,kind) in done: continue
            r=diagnostic(c,seed) if kind=='controlled_initial_goal' else run_episode(sc,seed,'llm_hibrido',c,1,4,60)
            r['summary'].update(diagnostic=kind,initial_goal='' if kind=='natural' else 'COLLECT')
            with log.open('a',encoding='utf8') as f: f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            rs.append(r);s=r['summary'];print(len(rs),'/10',c.model,sc,kind,'ações durante inferência:',s['actions_during_inference'],'frames:',s['frames'],'atrasos:',s['missed_frames'],flush=True)
    report=[]
    for r in rs:
        s=r['summary']; frames=r['frames']; iv=[f['interval_ms'] for f in frames if f['interval_ms']!='']; report.append(dict(model=s['model'],scenario=s['scenario'],seed=s['seed'],diagnostic=s['diagnostic'],success=s['success'],calls=s['calls'],actions_during_inference=s['actions_during_inference'],frames=len(frames),hz=1000/(sum(iv)/len(iv)) if iv else '',misses=s['missed_frames'],unsafe_commits=sum(f['commit'] and not f['safe'] for f in frames)))
    write_csv(folder/'diagnostico.csv',report);gate(folder,base)
if __name__=='__main__': main()
