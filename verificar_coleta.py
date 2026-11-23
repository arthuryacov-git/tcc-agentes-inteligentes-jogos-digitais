import argparse, collections, json, hashlib
from pathlib import Path
from experimento import load_records, PROTOCOL, observe, choices_for
from simulador import SCENARIOS, World

def main():
    p=argparse.ArgumentParser(); p.add_argument('folder'); p.add_argument('--gate',action='store_true'); a=p.parse_args(); folder=Path(a.folder)
    cfg=json.loads((folder/'config.json').read_text(encoding='utf8')); records=load_records(folder/'registros.jsonl'); groups=collections.defaultdict(list); seen=set(); errors=[]
    models={x['model']:x['digest'] for x in cfg['models']}
    for name,key in [('experimento.py','controller_sha256'),('simulador.py','world_sha256')]:
        if cfg.get(key)!=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest(): errors.append('Código não corresponde à configuração: '+name)
    for r in records:
        s=r['summary']; label=s['mode']+(':'+s['model'] if s['model'] else ''); key=(s['scenario'],s['seed'],label)
        if key in seen: errors.append('duplicado '+str(key))
        seen.add(key); groups[label].append(r)
        if s.get('protocol')!=PROTOCOL or len(r['rows'])!=s['turns'] or len(r['calls'])!=s['calls']: errors.append('registro inconsistente '+str(key))
        if not cfg['realtime'] and s['inference_waits']!=0: errors.append('espera por inferência em turno lógico '+str(key))
        if s['mode'].startswith('llm_'):
            if s['digest']!=models.get(s['model']) or not r['calls']: errors.append('identidade/inferência ausente '+str(key))
            for c in r['calls']:
                if c.get('model')!=s['model'] or c.get('digest')!=s['digest'] or c.get('tokens_in') is None or c.get('tokens_out') is None or c.get('ms',-1)<0: errors.append('metadados de inferência incompletos '+str(key)); break
        # Replay identical world seed and verify actual commands and all observations at logical boundaries.
        w=World(s['scenario'],s['seed']); observations={}
        for row in r['rows']:
            w.trigger(); o=observe(w,s['mode']=='llm_hibrido'); observations[w.t]=o
            if row['turn']!=w.t or row['location']!=w.loc or row['energy']!=w.energy or row['target']!=w.target: errors.append('estado antes da ação divergente '+str(key));break
            executed,result,_=w.step(row['action'])
            if executed!=row['executed'] or result!=row['result'] or row['location_after']!=w.loc or row['energy_after']!=w.energy or row['cargo']!=int(w.cargo) or row['accepted']!=int(w.accepted): errors.append('replay divergente '+str(key));break
        if w.t!=s['turns'] or int(w.success())!=s['success']: errors.append('resultado do replay diverge '+str(key))
        if not cfg['realtime']:
            for c in r['calls']:
                try:
                    payload=json.loads(c['prompt']['messages'][-1]['content']); obs=observations[c['request_turn']]
                    # JSON converts tuples to arrays; compare normalized representations.
                    if payload['observation']!=json.loads(json.dumps(obs)) or payload['allowed_choices']!=choices_for(obs,s['mode']=='llm_hibrido') or c['allowed']!=payload['allowed_choices']: raise ValueError('observação/opções')
                except (KeyError,ValueError,TypeError): errors.append('entrada de inferência difere do estado real '+str(key));break
        if any(f['commit'] and not f['safe'] for f in r['frames']): errors.append('commit fora de ponto seguro '+str(key))
    labels=['fsm_bt','planejador']+[m+':'+model for model in models for m in ('llm_puro','llm_hibrido')]
    expected={(sc,seed,label) for sc in SCENARIOS for seed in range(cfg['seed'],cfg['seed']+cfg['n']) for label in labels}
    print('Registros:',len(records),'/',len(expected),'| Ausentes:',len(expected-seen),'| Extras:',len(seen-expected))
    for label,rs in sorted(groups.items()):
        n=len(rs); summaries=[r['summary'] for r in rs]; calls=[c for r in rs for c in r['calls']]; steps=sum(s['turns'] for s in summaries); invalid=sum(s['invalid_outputs'] for s in summaries); waiting=sum(s['voluntary_waits'] for s in summaries)
        print(label, '| n=',n,'| sucessos=',sum(s['success'] for s in summaries),'| chamadas=',len(calls),'| saídas inválidas=',invalid,'| esperas voluntárias=',waiting,'/',steps,'| espera técnica=',sum(s['inference_waits'] for s in summaries),'| overrides=',sum(s['body_overrides'] for s in summaries),'| metas rejeitadas no contexto=',sum(s['context_rejected_turns'] for s in summaries),'| ações durante inferência=',sum(s['actions_during_inference'] for s in summaries))
        if calls and waiting==steps: print('  ALERTA: estagnação total; revisar observações e justificar antes de prosseguir. Não excluir por fracasso.')
        if invalid: print('  ALERTA: saídas inválidas são resultados registrados; examinar JSON bruto e regras.')
    if seen-expected: errors.append('registros fora do desenho')
    if a.gate and expected-seen: errors.append('amostra técnica incompleta')
    if a.gate and (cfg['n']!=2 or cfg['seed']!=20261001 or cfg['realtime'] or cfg['baseline_only'] or set(models)!={'qwen2.5:3b','llama3.2:3b'}): errors.append('configuração não é a verificação V5 prevista')
    if errors: raise SystemExit('\n'.join(errors))
    print('Integridade técnica OK. Sucesso comportamental não é condição de aprovação.')
if __name__=='__main__': main()
