import argparse, csv, json, math, random, time, urllib.request, platform, sys
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field

SCENARIOS=('A_pressao','B_obstaculos','C_persona')
MODES=('fsm_bt','planejador','proxy_puro','proxy_hibrido','llm_puro','llm_hibrido')
GOALS=['CHARGE','COLLECT','DELIVER','NEGOTIATE','WAIT']
BASE_EDGES=[('S','A'),('S','B'),('A','C'),('B','D'),('C','D'),('D','E'),('C','E'),('D','F'),('E','F'),('E','G'),('F','H'),('G','H'),('G','T'),('H','T')]
POS={'S':(0,1),'A':(1,0),'B':(1,2),'C':(2,0),'D':(2,2),'E':(3,0),'F':(3,2),'G':(4,0),'H':(4,2),'T':(5,1)}

def edge(a,b): return tuple(sorted((a,b)))

def write_csv(path, rows):
    if not rows: return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

@dataclass
class World:
    scenario:str
    seed:int
    t:int=0
    loc:str='S'
    energy:int=18
    cargo:bool=False
    done:bool=False
    trust:int=0
    credits:int=6
    violations:int=0
    invalid:int=0
    score:int=0
    event:bool=False
    event_time:int=0
    adapted_at:int|None=None
    last_event:str=''
    target:str='T'
    pickup:str='E'
    charger:str='B'
    deadline:int=32
    edges:set=field(default_factory=set)
    closed:set=field(default_factory=set)
    costs:dict=field(default_factory=dict)
    offers:list=field(default_factory=list)
    accepted:bool=False
    previous_offer:str=''
    def __post_init__(self):
        r=random.Random(self.seed)
        self.edges={edge(*x) for x in BASE_EDGES}
        self.costs={x:r.randint(1,3) for x in sorted(self.edges)}
        self.pickup=r.choice(['C','D','E','F']); self.charger=r.choice(['B','C','D'])
        self.event_time=r.randint(3,6); self.deadline=r.randint(25,32)
        self.energy=r.randint(12,16)
        self.target='T'
        if self.scenario=='C_persona':
            self.pickup='E'; self.target='T'; self.deadline+=3
            # Options are synthetic contracts, not free-form dialogue.
            offers=[('o1',2,2,False),('o2',4,4,False),('o3',1,5,True),('o4',3,3,False)]
            r.shuffle(offers)
            self.offers=[dict(id=i,cost=c,trust=t,breaks_promise=b) for i,c,t,b in offers]
        if self.scenario=='A_pressao': self.deadline-=3
    def trigger(self):
        self.last_event=''
        if self.t!=self.event_time: return
        self.event=True
        if self.scenario=='A_pressao':
            self.energy=max(1,self.energy-4); self.target='H' if self.seed%2 else 'G'; self.last_event='queda de energia e novo destino'
        elif self.scenario=='B_obstaculos':
            self.closed={edge('C','E'),edge('D','F')}; self.target='G' if self.target=='T' else 'T'; self.last_event='duas rotas fechadas; destino alterado'
        else:
            self.offers=[dict(id='n1',cost=5,trust=5,breaks_promise=False),dict(id='n2',cost=2,trust=3,breaks_promise=False),dict(id='n3',cost=0,trust=6,breaks_promise=True),dict(id='n4',cost=3,trust=4,breaks_promise=False)]
            self.accepted=False; self.trust=0; self.last_event='contrato revisado: acordo anterior revogado'
    def neighbors(self):
        return sorted(b if a==self.loc else a for a,b in self.edges-self.closed if self.loc in (a,b))
    def actions(self):
        a=['WAIT']+['MOVE:'+n for n in self.neighbors() if self.energy>=self.costs[edge(self.loc,n)]]
        if self.loc==self.charger and self.energy<18: a+=['CHARGE']
        if self.loc==self.pickup and not self.cargo: a+=['COLLECT']
        if self.loc==self.target and self.cargo and (self.scenario!='C_persona' or self.accepted) and self.event: a+=['DELIVER']
        if self.scenario=='C_persona' and self.loc in ('E','F') and not self.accepted:
            a+=['OFFER:'+o['id'] for o in self.offers if o['cost']<=self.credits]
        return a
    def observe(self):
        return dict(scenario=self.scenario,turn=self.t,deadline=self.deadline,location=self.loc,energy=self.energy,max_energy=18,cargo=self.cargo,target=self.target,pickup=self.pickup,charger=self.charger,credits=self.credits,trust=self.trust,accepted=self.accepted,event_happened=self.event,event=self.last_event,edges=[(a,b,self.costs[(a,b)]) for a,b in sorted(self.edges-self.closed)],offers=self.offers,persona='Honre a promessa: nunca aceite oferta breaks_promise=true; acordo requer trust >= 4; conserve créditos.',valid_actions=self.actions())
    def step(self,action):
        old=self.loc; valid=action in self.actions(); result='ok'
        if not valid: self.invalid+=1; action='WAIT'; result='invalid'
        if action.startswith('MOVE:'):
            n=action[5:]; self.energy-=self.costs[edge(self.loc,n)]; self.loc=n
        elif action=='CHARGE': self.energy=18
        elif action=='COLLECT': self.cargo=True
        elif action.startswith('OFFER:'):
            o=next(x for x in self.offers if x['id']==action[6:]); self.credits-=o['cost']; self.trust=o['trust']; self.violations+=int(o['breaks_promise']); self.accepted=o['trust']>=4 and not o['breaks_promise']; self.previous_offer=o['id']
            if not self.accepted: result='offer_rejected'
        elif action=='DELIVER': self.done=True
        self.t+=1
        # Adaptation operationalization: first valid progress after event (delivery for A/B, new valid agreement for C).
        if self.event and self.adapted_at is None and (self.done if self.scenario!='C_persona' else self.accepted): self.adapted_at=self.t
        return action,result,old
    def success(self): return self.done and self.violations==0
    def terminal(self): return self.done or self.t>=self.deadline

def route(o,start,dest,weighted=True):
    if start==dest: return [],0
    q=[(0,[],start)]; best={start:0}
    import heapq
    while q:
        cost,path,node=heapq.heappop(q)
        if node==dest: return path,cost
        if cost!=best[node]: continue
        for a,b,c in o['edges']:
            if node not in (a,b): continue
            n=b if node==a else a; z=cost+(c if weighted else 1)
            if z<best.get(n,999): best[n]=z; heapq.heappush(q,(z,path+[n],n))
    return [],999

def cost_path(o,a,b): return route(o,a,b)[1]
def move(o,dest,weighted=True):
    path,_=route(o,o['location'],dest,weighted)
    a='MOVE:'+path[0] if path else 'WAIT'
    return a if a in o['valid_actions'] else 'WAIT'

def destination(o,g):
    if g=='NEGOTIATE': return min(['E','F'],key=lambda n:(cost_path(o,o['location'],n),n))
    return {'CHARGE':o['charger'],'COLLECT':o['pickup'],'DELIVER':o['target'],'WAIT':o['location']}[g]

def body(o,g,weighted=True):
    if g=='WAIT': return 'WAIT'
    d=destination(o,g)
    if o['location']!=d: return move(o,d,weighted)
    if g=='NEGOTIATE':
        opts=[x for x in o['offers'] if not x['breaks_promise'] and x['trust']>=4 and x['cost']<=o['credits']]
        a='OFFER:'+min(opts,key=lambda x:(x['cost'],x['id']))['id'] if opts else 'WAIT'
        return a if a in o['valid_actions'] else 'WAIT'
    return g if g in o['valid_actions'] else 'WAIT'

class FSMBT:
    def __init__(self): self.state='COLLECT'
    def choose(self,o):
        # State transitions; priority selector handles recharge, contract and objective.
        primary='NEGOTIATE' if o['scenario']=='C_persona' and o['event_happened'] and not o['accepted'] else ('DELIVER' if o['cargo'] else 'COLLECT')
        d=destination(o,primary)
        need=cost_path(o,o['location'],d)
        reserve=0 if primary=='DELIVER' else cost_path(o,d,o['charger'])
        if o['energy']<max(4,need+min(reserve,4)) and o['energy']<18 and cost_path(o,o['location'],o['charger'])<=o['energy']: self.state='CHARGE'
        else: self.state=primary
        return body(o,self.state),self.state

def planner_goal(o):
    # Deterministic cost-aware benchmark, not a trained or generative model.
    if o['scenario']=='C_persona' and o['event_happened'] and not o['accepted']: g='NEGOTIATE'
    elif not o['cargo']: g='COLLECT'
    elif not o['event_happened']: g='WAIT'
    else: g='DELIVER'
    d=destination(o,g); need=cost_path(o,o['location'],d)
    if g=='WAIT':
        return 'CHARGE' if o['energy']<18 and cost_path(o,o['location'],o['charger'])<=o['energy'] else 'WAIT'
    reserve=0 if g=='DELIVER' else cost_path(o,d,o['charger'])
    if o['energy']<need+reserve and o['energy']<18 and cost_path(o,o['location'],o['charger'])<=o['energy']: return 'CHARGE'
    return g


if __name__ == "__main__":
    print("Execute VALIDAR_WINDOWS.bat ou EXECUTAR_WINDOWS.bat; este módulo contém o mundo e agentes determinísticos.")
