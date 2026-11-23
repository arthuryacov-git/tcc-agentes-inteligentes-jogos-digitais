import os, random, argparse, csv, hashlib, json, math, platform, sys, time, urllib.request, threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from simulador import World, FSMBT, planner_goal, body, destination, cost_path, GOALS, SCENARIOS, write_csv

PROTOCOL = 'V5.0'

def goal_status(o,g):
    if g=='WAIT': return 'available'
    if g=='COLLECT': return 'completed' if o['cargo'] else 'available'
    if g=='CHARGE': return 'completed' if o['energy']==o['max_energy'] else 'available'
    if g=='NEGOTIATE':
        if o['scenario']!='C_persona': return 'not_applicable'
        return 'completed' if o.get('accepted',False) else 'available'
    if g=='DELIVER':
        if not o['cargo']: return 'requires_cargo'
        if o['scenario']=='C_persona' and not o.get('accepted',False): return 'requires_agreement'
        return 'available'
    return 'unknown'

def choices_for(o, hybrid):
    return [g for g in GOALS if goal_status(o,g)=='available'] if hybrid else list(o['valid_actions'])

def system_prompt(o, hybrid):
    rules = ('Primary objective: deliver the cargo before the deadline, AFTER the scheduled world event, without violating the persona if applicable. '
             'Every action including WAIT consumes one logical turn. Energy is a resource, not the objective. '
             'The map is an undirected graph: each row of adjacency lists reachable neighbors and movement energy cost, not a recommended route. '
             'MOVE:X travels ONE open edge; CHARGE restores energy to 18 at charger; COLLECT picks up cargo at pickup; DELIVER completes delivery at target. '
             'Use mission_status to identify completed and pending requirements. A cargo_collected value of true means collection is already complete. '
             'The event may change the world; its future timing is unknown. You can make progress before it occurs. WAIT is allowed but consumes the deadline. '
             'Use recent executed actions and outcomes as memory; the previous intention may be obsolete. ')
    if o['scenario']=='A_pressao': rules += 'Scenario A: event may reduce energy and change destination. No negotiation. '
    elif o['scenario']=='B_obstaculos': rules += 'Scenario B: event may close routes and change destination. No negotiation. '
    else: rules += ('Scenario C: delivery requires an accepted agreement, and the event revokes any previous agreement. '
                    'Offers at E or F cost credits; trust >= 4 and no broken promise are required. A promise violation makes the episode unsuccessful. ')
    if hybrid: rules += ('Choose a strategic goal from allowed_choices; the Body chooses the primitive action. '
                         'COLLECT seeks pickup, DELIVER seeks target (can position before event), CHARGE seeks charger, NEGOTIATE seeks agreement, WAIT holds position. '
                         'Completed or inapplicable goals and goals missing mandatory prerequisites are excluded. The Body may recharge as a safety override. ')
    else: rules += 'Choose a primitive action from allowed_choices; plan multiple steps mentally, but return one action only. '
    return rules+'Return JSON with action (exact allowed choice) and note (short intention, at most 20 words). Do not output extra text.'

@dataclass(frozen=True)
class Message:
    seq: int
    epoch: int
    value: str

class Blackboard:
    def __init__(self):
        self.lock=threading.Lock(); self.pending=None; self.active=None; self.latest=-1
        self.rejected=0; self.stale=0; self.commits=0; self.overwritten=0
    def publish(self,m):
        with self.lock:
            if m.seq<=self.latest:
                self.rejected+=1; return False
            self.latest=m.seq
            if self.pending is not None: self.overwritten+=1
            self.pending=m; return True
    def promote(self,safe,epoch):
        with self.lock:
            if safe:
                if self.active and self.active.epoch!=epoch: self.active=None
                if self.pending:
                    m=self.pending; self.pending=None
                    if m.epoch!=epoch: self.stale+=1
                    else: self.active=m; self.commits+=1
            return self.active
    def snapshot(self):
        with self.lock: return self.active,self.pending

class Ollama:
    def __init__(self,url,model): self.url=url.rstrip('/'); self.model=model; self.digest=''; self.version=''
    def req(self,path,payload=None):
        data=None if payload is None else json.dumps(payload).encode()
        req=urllib.request.Request(self.url+path,data=data,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=180) as r: return json.load(r)
    def preflight(self):
        self.version=self.req('/api/version')['version']; models=self.req('/api/tags')['models']; match=[m for m in models if m.get('name')==self.model or m.get('model')==self.model]
        if not match: raise RuntimeError('Modelo não instalado: '+self.model)
        self.digest=match[0]['digest']; return dict(model=self.model,digest=self.digest,ollama=self.version)
    def infer(self,o,choices,seed,previous):
        payload={'model':self.model,'stream':False,'keep_alive':'30m','format':{'type':'object','properties':{'action':{'type':'string','enum':choices},'note':{'type':'string'}},'required':['action','note']},'options':{'temperature':0.2,'seed':seed,'num_predict':128, 'num_ctx':4096},'messages':[{'role':'system','content':system_prompt(o, bool(o.get('strategic_mode',False)))},{'role':'user','content':json.dumps({'observation':o,'allowed_choices':choices,'previous_intention':previous},ensure_ascii=False)}]}
        t=time.perf_counter(); response=self.req('/api/chat',payload); elapsed=(time.perf_counter()-t)*1000
        raw=response['message']['content']; action='INVALID_OUTPUT'; note=''; parse_error=False
        try:
            d=json.loads(raw); action=str(d['action']); note=str(d['note'])[:200]
            if not isinstance(d['action'],str) or not isinstance(d['note'],str): raise TypeError('schema')
        except (ValueError,TypeError,KeyError): parse_error=True; action='INVALID_OUTPUT'
        return dict(action=action,note=note,raw=raw,ms=elapsed,tokens_in=response.get('prompt_eval_count'),tokens_out=response.get('eval_count'),server_ms=response.get('total_duration',0)/1e6,load_ms=response.get('load_duration',0)/1e6,eval_ms=response.get('eval_duration',0)/1e6,parse_error=parse_error,done_reason=response.get('done_reason',''),prompt=payload,model=self.model,digest=self.digest)

def action_body(o,active):
    g=active or 'WAIT'
    if g not in ('WAIT','CHARGE') and o['energy']<cost_path(o,o['location'],destination(o,g)) and cost_path(o,o['location'],o['charger'])<=o['energy']: g='CHARGE'
    return body(o,g),g

def run_realtime(scenario,seed,mode,client=None,tick_s=1.0,period=4,frame_hz=60,bootstrap_goal=None):
    w=World(scenario,seed); fsm=FSMBT(); bb=Blackboard(); hybrid=mode=='llm_hibrido'; real=mode.startswith('llm_'); pure=mode=='llm_puro'
    calls=[]; rows=[]; frames=[]; body_times=[]; previous={}; future=None; seq=0; next_req=0; last_seq=None; lastmove=None; reversals=0
    start=time.perf_counter(); cpu=time.process_time(); cadence=1/frame_hz; tick_frames=max(1,round(tick_s*frame_hz)); frame=0; last_frame=None
    if bootstrap_goal is not None:
        if not hybrid or bootstrap_goal not in choices_for(observe(w,True),True): raise ValueError('Invalid diagnostic bootstrap')
        bb.publish(Message(0,0,bootstrap_goal)); bb.promote(True,0); seq=1
    def worker(obs,choices,s,e,prev):
        st=time.perf_counter(); result=client.infer(obs,choices,seed+obs['turn'],prev)
        result.update(seq=s,epoch=e,request_turn=obs['turn'],allowed=choices,worker_start_ms=(st-start)*1000,worker_finish_ms=(time.perf_counter()-start)*1000)
        bb.publish(Message(s,e,result['action'] if not result['parse_error'] and result['action'] in choices else 'INVALID_OUTPUT')); return result
    def receive():
        nonlocal future,previous
        r=future.result(); r['received_turn']=w.t; calls.append(r); previous=dict(action=r['action'],note=r['note'],request_turn=r['request_turn']); future=None
    def submit(o):
        nonlocal future,seq,next_req
        future=pool.submit(worker,o,choices_for(o,hybrid),seq,int(w.event),memory(previous,rows)); seq+=1; next_req=w.t+period
    with ThreadPoolExecutor(max_workers=1) as pool:
        while not w.terminal():
            target=start+frame*cadence; remain=target-time.perf_counter()
            if remain>0: time.sleep(remain)
            begin=time.perf_counter(); safe=frame%tick_frames==0; executed=''; commit=False
            if safe: w.trigger()
            o=observe(w,hybrid); full=w.observe(); epoch=int(w.event)
            if future is not None and future.done(): receive()
            old,_=bb.snapshot(); msg=bb.promote(safe,epoch); commit=msg is not None and msg!=old
            active=msg is not None and msg.epoch==epoch
            # Pure requests after acting; hybrid schedules periodic refresh before a body tick.
            # No sleeping for inference, no blocking on future.result() except after termination.
            if real and future is None:
                _,pending=bb.snapshot()
                if pending is None:
                    if pure and (not active or msg.seq==last_seq): submit(o)
                    elif hybrid:
                        due=not active or msg.value not in choices_for(o,True) or w.t>=next_req
                        lead=(frame%tick_frames==max(0,tick_frames-4))
                        if due and (not active or lead or safe and msg.value not in choices_for(o,True) or frame==0): submit(o)
            a=time.perf_counter(); goal=''; action='WAIT'; requested=''; source=mode; goal_seq=''
            if mode=='fsm_bt': action,goal=fsm.choose(full)
            elif mode=='planejador': goal=planner_goal(full); action=body(full,goal)
            elif hybrid:
                source='inference_wait'
                if active:
                    requested=msg.value; goal_seq=msg.seq
                    if requested not in choices_for(o,True):
                        source='context_rejected' if requested in GOALS else 'invalid_model_goal'; action='WAIT' if requested in GOALS else 'INVALID_OUTPUT'
                    else:
                        action,goal=action_body(full,requested); source='body_override' if goal!=requested else 'body_goal'
            elif pure:
                source='inference_wait'
                if active and msg.seq!=last_seq:
                    action=requested=msg.value; goal_seq=msg.seq; source='llm_action'
            dt=(time.perf_counter()-a)*1000; body_times.append(dt)
            if safe:
                if source=='llm_action': last_seq=msg.seq
                before=w.loc; executed,result,_=w.step(action); at=(time.perf_counter()-start)*1000
                if executed.startswith('MOVE:'):
                    if lastmove==(w.loc,before) and not o['event']: reversals+=1
                    lastmove=(before,w.loc)
                else: lastmove=None
                rows.append(dict(scenario=scenario,seed=seed,mode=mode,turn=o['turn'],location=before,energy=o['energy'],target=o['target'],pickup=o['pickup'],charger=o['charger'],event=o['event'],goal=goal,requested=requested,source=source,goal_seq=goal_seq,action=action,executed=executed,result=result,location_after=w.loc,energy_after=w.energy,cargo=int(w.cargo),accepted=int(w.accepted),credits=w.credits,closed=json.dumps(sorted(w.closed)),decision_ms='',body_ms=dt,note='',raw='',action_time_ms=at,inference_pending=int(future is not None and not future.done()),goal_context=goal_status(o,requested) if hybrid and requested else ''))
            end=time.perf_counter(); frames.append(dict(frame=frame,turn=w.t,safe=int(safe),commit=int(commit),interval_ms='' if last_frame is None else (begin-last_frame)*1000,work_ms=(end-begin)*1000,late_ms=(begin-target)*1000,miss=int(end>target+cadence),action=executed))
            last_frame=begin; frame+=1
        elapsed=(time.perf_counter()-start)*1000
        if future is not None:
            r=future.result(); r['received_turn']=None; calls.append(r)
    avg=lambda x:sum(x)/len(x) if x else None
    return dict(summary=dict(scenario=scenario,seed=seed,mode=mode,success=int(w.success()),delivered=int(w.done),turns=w.t,deadline=w.deadline,adapted=int(w.adapted_at is not None),adaptation_turns='' if w.adapted_at is None else w.adapted_at-w.event_time,persona_violations=w.violations,invalid_actions=w.invalid,reversals=reversals,energy_left=w.energy,credits_left=w.credits,calls=len(calls),tokens_in=sum(c['tokens_in'] for c in calls) if real and all(c['tokens_in'] is not None for c in calls) else '',tokens_out=sum(c['tokens_out'] for c in calls) if real and all(c['tokens_out'] is not None for c in calls) else '',decision_ms=avg([c['ms'] for c in calls]) if real else avg(body_times),body_ms=avg(body_times),cpu_ms=(time.process_time()-cpu)*1000,wall_ms=elapsed,virtual_delay='',tick_ms=tick_s*1000,handoffs=bb.commits,stale_responses=bb.stale,waiting_ticks=0,model=client.model if real else '',digest=client.digest if real else '',invalid_outputs=sum(int(c['parse_error'] or c['action'] not in c['allowed']) for c in calls),frame_hz=frame_hz,missed_frames=sum(r['miss'] for r in frames),frames=len(frames)),rows=rows,calls=calls,frames=frames)

def observe(w,hybrid=False):
    o=w.observe(); o['strategic_mode']=hybrid
    o['turns_remaining']=o['deadline']-o['turn']
    if w.scenario!='C_persona':
        for k in ('offers','persona','credits','trust','accepted'): o.pop(k,None)
    o['adjacency']={n:[] for n in sorted({v for a,b,c in o['edges'] for v in (a,b)})}
    for a,b,c in o['edges']:
        o['adjacency'][a].append({'to':b,'energy_cost':c}); o['adjacency'][b].append({'to':a,'energy_cost':c})
    o['mission_status']={'cargo_collected':w.cargo,'event_occurred':w.event,'agreement_required':w.scenario=='C_persona','agreement_accepted':w.accepted if w.scenario=='C_persona' else None,'delivered':w.done,'at_pickup':w.loc==w.pickup,'at_target':w.loc==w.target,'at_charger':w.loc==w.charger,'delivery_available':'DELIVER' in w.actions()}
    if hybrid: o['goal_status']={g:goal_status(o,g) for g in GOALS}
    return o

def goal_done(o,g): return goal_status(o,g)=='completed'

def memory(previous,rows):
    return {**(previous if isinstance(previous,dict) else {}),'recent_actions':[{'turn':r['turn'],'executed':r['executed'],'result':r['result'],'location_after':r['location_after'],'energy_after':r['energy_after'],'cargo':r['cargo'],'accepted':r['accepted']} for r in rows[-4:]]}

def run_logical(scenario,seed,mode,client=None,period=4):
    w=World(scenario,seed); fsm=FSMBT(); bb=Blackboard(); hybrid=mode=='llm_hibrido'; real=mode.startswith('llm_')
    calls=[]; rows=[]; body_times=[]; previous={}; seq=0; next_req=0; future=None; last_result=None
    start=time.perf_counter(); cpu=time.process_time(); lastmove=None; reversals=0
    def worker(obs,choices,s,e,prev):
        worker_start=time.perf_counter(); r=client.infer(obs,choices,seed+obs['turn'],prev); r.update(seq=s,epoch=e,request_turn=obs['turn'],allowed=choices,worker_start_ms=(worker_start-start)*1000,worker_finish_ms=(time.perf_counter()-start)*1000)
        if hybrid: bb.publish(Message(s,e,r['action'] if not r['parse_error'] and r['action'] in choices else 'INVALID_OUTPUT'))
        return r
    def receive(f):
        nonlocal previous,last_result
        r=f.result(); r['received_turn']=w.t; calls.append(r); last_result=r
        previous=dict(action=r['action'],note=r['note'],request_turn=r['request_turn'])
        return r
    with ThreadPoolExecutor(max_workers=1) as pool:
        while not w.terminal():
            w.trigger(); o=observe(w,hybrid); full=w.observe(); epoch=int(w.event)
            if future is not None:
                # A fixed one-turn handoff, never conditional on real inference speed.
                r=receive(future); future=None
                r['context_at_handoff']=goal_status(o,r['action']) if hybrid else ''
            msg=bb.promote(True,epoch); initial=msg is None
            if real:
                refresh=not hybrid or initial or w.t>=next_req or goal_done(o,msg.value) or msg.value not in choices_for(o,hybrid)
                if refresh:
                    choices=choices_for(o,hybrid); prev=memory(previous,rows)
                    future=pool.submit(worker,o,choices,seq,epoch,prev); seq+=1; next_req=w.t+period
                    # Initial/event/completed-goal refresh must be available before acting.
                    # Periodic refresh overlaps exactly one logical action using the active goal.
                    if not hybrid or initial or goal_done(o,msg.value) or msg.value not in choices:
                        r=receive(future); future=None; r['context_at_handoff']=goal_status(o,r['action']) if hybrid else ''; msg=bb.promote(True,epoch)
            a=time.perf_counter(); goal=''; requested=''; source=mode; note=''; raw=''; decision_ms=''; goal_seq=''
            if mode=='fsm_bt': action,goal=fsm.choose(full)
            elif mode=='planejador': goal=planner_goal(full); action=body(full,goal)
            elif hybrid:
                requested=msg.value if msg else 'INVALID_OUTPUT'; goal_seq=msg.seq if msg else ''
                if requested not in choices_for(o,True): action='WAIT' if requested in GOALS else 'INVALID_OUTPUT'; goal=''; source='context_rejected' if requested in GOALS else 'invalid_model_goal'
                else:
                    action,goal=action_body(full,requested); source='body_override' if goal!=requested else 'body_goal'
                if last_result and msg and last_result['seq']==msg.seq: note=last_result['note']; raw=last_result['raw']; decision_ms=last_result['ms']
            else:
                requested=last_result['action']; action=requested if not last_result['parse_error'] and requested in last_result['allowed'] else 'INVALID_OUTPUT'; source='llm_action'; note=last_result['note']; raw=last_result['raw']; decision_ms=last_result['ms']; goal_seq=last_result['seq']
            dt=(time.perf_counter()-a)*1000; body_times.append(dt)
            before=w.loc; executed,result,_=w.step(action)
            if executed.startswith('MOVE:'):
                if lastmove==(w.loc,before) and not o['event']: reversals+=1
                lastmove=(before,w.loc)
            else: lastmove=None
            rows.append(dict(scenario=scenario,seed=seed,mode=mode,turn=o['turn'],location=before,energy=o['energy'],target=o['target'],pickup=o['pickup'],charger=o['charger'],event=o['event'],goal=goal,requested=requested,source=source,goal_seq=goal_seq,action=action,executed=executed,result=result,location_after=w.loc,energy_after=w.energy,cargo=int(w.cargo),accepted=int(w.accepted),credits=w.credits,closed=json.dumps(sorted(w.closed)),decision_ms=decision_ms,body_ms=dt,note=note,raw=raw,action_time_ms=(time.perf_counter()-start)*1000,inference_pending=int(future is not None and not future.done()),goal_context=goal_status(o,requested) if hybrid and requested else ''))
        elapsed=(time.perf_counter()-start)*1000
        if future is not None:
            r=future.result(); r['received_turn']=None; calls.append(r)
    avg=lambda x:sum(x)/len(x) if x else None
    summary=dict(scenario=scenario,seed=seed,mode=mode,success=int(w.success()),delivered=int(w.done),turns=w.t,deadline=w.deadline,adapted=int(w.adapted_at is not None),adaptation_turns='' if w.adapted_at is None else w.adapted_at-w.event_time,persona_violations=w.violations,invalid_actions=w.invalid,reversals=reversals,energy_left=w.energy,credits_left=w.credits,calls=len(calls),tokens_in=sum(c['tokens_in'] for c in calls) if real and all(c['tokens_in'] is not None for c in calls) else '',tokens_out=sum(c['tokens_out'] for c in calls) if real and all(c['tokens_out'] is not None for c in calls) else '',decision_ms=avg([c['ms'] for c in calls]) if real else avg(body_times),body_ms=avg(body_times),cpu_ms=(time.process_time()-cpu)*1000,wall_ms=elapsed,virtual_delay='',tick_ms='',handoffs=bb.commits,stale_responses=bb.stale,waiting_ticks=0,model=client.model if real else '',digest=client.digest if real else '',invalid_outputs=sum(int(c['parse_error'] or c['action'] not in c['allowed']) for c in calls),frame_hz='',missed_frames=0,frames=0)
    return dict(summary=summary,rows=rows,calls=calls,frames=[])

def run_episode(scenario,seed,mode,client=None,tick_s=1.0,period=4,frame_hz=None):
    r=run_realtime(scenario,seed,mode,client,tick_s,period,frame_hz) if frame_hz else run_logical(scenario,seed,mode,client,period)
    rows=r['rows']; s=r['summary']; streak=0; longest=0; seen=set(); revisits=0; no_change=0
    for row in rows:
        wait=row['executed']=='WAIT'; streak=streak+1 if wait else 0; longest=max(longest,streak)
        sig=(row['location_after'],row['energy_after'],row['cargo'],row['accepted'],row['credits'],row['target'],row['closed'])
        if sig in seen: revisits+=1
        seen.add(sig)
        if row['location']==row['location_after'] and row['energy']==row['energy_after'] and wait: no_change+=1
    used={x['goal_seq'] for x in rows if x.get('goal_seq')!='' and x.get('source') in ('llm_action','body_goal','body_override')}
    s.update(protocol=PROTOCOL,backend='ollama' if client else 'deterministic',wait_actions=sum(x['executed']=='WAIT' for x in rows),voluntary_waits=sum(x['action']=='WAIT' and (x.get('source') in ('llm_action','fsm_bt','planejador') or x.get('source')=='body_goal' and x['requested']=='WAIT') for x in rows),body_blocked_waits=sum(x['action']=='WAIT' and x.get('source') in ('body_goal','body_override') and x['requested']!='WAIT' for x in rows),invalid_waits=sum(x['result']=='invalid' for x in rows),inference_waits=sum(x.get('source')=='inference_wait' for x in rows),longest_wait_streak=longest,revisited_states=revisits,wait_no_change=no_change,body_overrides=sum(x.get('source')=='body_override' for x in rows),body_goal_actions=sum(x.get('source')=='body_goal' for x in rows),llm_direct_actions=sum(x.get('source')=='llm_action' for x in rows),used_responses=len(used),actions_during_inference=sum(x['inference_pending'] and x['executed']!='WAIT' for x in rows))
    # Compare actual timestamps after all requests have finished. Pending future != actual worker overlap.
    for call in r['calls']:
        call.setdefault('context_at_handoff','')
    for row in rows:
        at=row['action_time_ms']; row['actual_inference_overlap']=int(any(c['worker_start_ms']<=at<c['worker_finish_ms'] for c in r['calls']))
    s['actions_during_inference']=sum(x['actual_inference_overlap'] and x['executed']!='WAIT' for x in rows)
    s['scheduled_overlap_actions']=sum(x['inference_pending'] and x['executed']!='WAIT' for x in rows)
    s['context_rejected_turns']=sum(x['source']=='context_rejected' for x in rows)
    s['redundant_at_handoff']=sum(c.get('context_at_handoff')=='completed' for c in r['calls'])
    s['redundant_goal_turns']=sum(x.get('goal_context')=='completed' for x in rows)
    s['redundant_goal_responses']=sum(c['action'] in GOALS and c.get('prompt',{}).get('messages') and json.loads(c['prompt']['messages'][-1]['content'])['observation'].get('goal_status',{}).get(c['action'])=='completed' for c in r['calls']) if client and all('prompt' in c for c in r['calls']) else 0
    s['waiting_ticks']=s['inference_waits']
    return r

def load_records(path):
    if not path.exists(): return []
    with path.open(encoding='utf8') as f:
        records=[]
        for i,line in enumerate(f,1):
            if not line.strip(): continue
            try: records.append(json.loads(line))
            except ValueError as e: raise RuntimeError(f'JSONL inválido na linha {i}; preserve uma cópia antes de reparar.') from e
        return records

def export(path,records):
    summaries=[]; decisions=[]; calls=[]; frames=[]
    for r in records:
        s=r['summary'].copy(); label=s['mode']+(':'+s['model'] if s['model'] else ''); s['mode']=label; summaries.append(s)
        for x in r['rows']: decisions.append({**x,'mode':label})
        for x in r['calls']: calls.append(dict(scenario=s['scenario'],seed=s['seed'],mode=label,**{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in x.items()}))
        for x in r['frames']: frames.append(dict(scenario=s['scenario'],seed=s['seed'],mode=label,**x))
    write_csv(path/'episodios.csv',summaries); write_csv(path/'decisoes.csv',decisions); write_csv(path/'inferencias.csv',calls); write_csv(path/'frames.csv',frames)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--models',nargs='+',default=['qwen2.5:3b','llama3.2:3b']); p.add_argument('--url',default='http://127.0.0.1:11434'); p.add_argument('--n',type=int,default=200); p.add_argument('--seed',type=int,default=20261119); p.add_argument('--tick-s',type=float,default=1.0); p.add_argument('--period',type=int,default=4); p.add_argument('--out',default='resultados/v5_principal'); p.add_argument('--realtime',action='store_true'); p.add_argument('--baseline-only',action='store_true'); p.add_argument('--export-only',action='store_true'); p.add_argument('--warmups',type=int,default=2); p.add_argument('--verified-config'); args=p.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True); log=out/'registros.jsonl'; cfg=out/'config.json'; records=load_records(log)
    if args.export_only: export(out,records); return
    if args.n<1 or args.warmups<2 or args.tick_s<=0 or args.period<1 or len(set(args.models))!=len(args.models): p.error('Parâmetros inválidos')
    clients=[]; meta=[]
    if not args.baseline_only:
        for m in args.models:
            c=Ollama(args.url,m); meta.append(c.preflight()); clients.append(c)
            if c.digest.upper().startswith('MOCK') or c.version.upper().startswith('MOCK'): raise RuntimeError('Servidor de teste não permitido na coleta')
    settings=dict(protocol=PROTOCOL, url=args.url, warmups=args.warmups, temperature=0.2,num_predict=128,num_ctx=4096,hybrid_overlap_turns=1,goal_filter="context_v1",n=args.n,seed=args.seed,tick_s=args.tick_s,period=args.period,realtime=args.realtime,baseline_only=args.baseline_only,models=meta,python=sys.version,platform=platform.platform(),controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),world_sha256=hashlib.sha256(Path(__file__).with_name('simulador.py').read_bytes()).hexdigest())
    if args.verified_config:
        v=json.loads(Path(args.verified_config).read_text(encoding='utf-8'))
        for k in ('protocol','models','controller_sha256','world_sha256','temperature','num_predict','num_ctx','period','warmups','hybrid_overlap_turns','goal_filter','url'):
            if v.get(k)!=settings.get(k): raise RuntimeError('Configuração difere da verificação: '+k)
    if cfg.exists() and json.loads(cfg.read_text(encoding='utf-8'))!=settings: raise RuntimeError('Configuração mudou; use uma nova pasta.')
    if not cfg.exists():
        tmp=cfg.with_suffix('.tmp'); tmp.write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf8'); os.replace(tmp,cfg)
    if len(records)!=len({(r['summary']['scenario'],r['summary']['seed'],r['summary']['mode'],r['summary']['model']) for r in records}): raise RuntimeError('Registros duplicados')
    if any(r['summary'].get('protocol')!=PROTOCOL for r in records): raise RuntimeError('Dados de outra versão')
    for c in clients:
        for j in range(args.warmups):
            o=observe(World('A_pressao',args.seed-100-j),False); r=c.infer(o,choices_for(o,False),args.seed-100-j,{})
            r.update(session_time=time.time(),warmup=j,protocol=PROTOCOL)
            with (out/'aquecimento.jsonl').open('a',encoding='utf8') as f: f.write(json.dumps(r,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
            if r['parse_error'] or r['action'] not in choices_for(o,False): raise RuntimeError('Falha de formato no aquecimento: '+c.model)
    done={(r['summary']['scenario'],r['summary']['seed'],r['summary']['mode'],r['summary']['model']) for r in records}
    session_time=time.time()
    try:
        # Pair by seed, alternate execution order to reduce fixed-order bias.
        for i in range(args.n):
            seed=args.seed+i
            for sc in SCENARIOS:
                jobs=[('fsm_bt',None),('planejador',None)]+[(mode,c) for c in clients for mode in ('llm_puro','llm_hibrido')]
                random.Random(seed*31+SCENARIOS.index(sc)).shuffle(jobs)
                for mode,c in jobs:
                    key=(sc,seed,mode,c.model if c else '')
                    if key in done: continue
                    r=run_episode(sc,seed,mode,c,args.tick_s,args.period,60 if args.realtime else None)
                    r['summary']['session_time']=session_time
                    with log.open('a',encoding='utf8') as f: f.write(json.dumps(r,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())
                    records.append(r); done.add(key); print(f"{len(records)}/{args.n*len(SCENARIOS)*(2+2*len(clients))}",sc,seed,mode,key[-1], 'sucesso='+str(r['summary']['success']), 'turnos='+str(r['summary']['turns']), 'chamadas='+str(r['summary']['calls']),flush=True)
    finally: export(out,records)
if __name__=='__main__': main()
