import unittest,time,threading,json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from experimento import Ollama,run_episode,Blackboard,Message
from simulador import planner_goal,body,World,SCENARIOS
from experimento import system_prompt,observe,choices_for

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def output(self,o):
        b=json.dumps(o).encode(); self.send_response(200); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self.output({'version':'MOCK'} if self.path=='/api/version' else {'models':[{'name':'mock:only','digest':'MOCK-NOT-LLM'}]})
    def do_POST(self):
        p=json.loads(self.rfile.read(int(self.headers['Content-Length']))); d=json.loads(p['messages'][-1]['content']); o=d['observation']; choices=d['allowed_choices']; goal=planner_goal(o); action=goal if o['strategic_mode'] else body(o,goal); action=action if action in choices else 'WAIT'; time.sleep(.035)
        self.output({'message':{'content':json.dumps({'action':action,'note':'mock'})},'prompt_eval_count':0,'eval_count':0})

class Tests(unittest.TestCase):
    def test_blackboard(self):
        b=Blackboard(); b.publish(Message(0,0,'COLLECT')); self.assertIsNone(b.promote(False,0)); self.assertEqual(b.promote(True,0).value,'COLLECT'); b.publish(Message(1,0,'DELIVER')); self.assertEqual(b.promote(False,1).value,'COLLECT'); self.assertIsNone(b.promote(True,1)); self.assertEqual(b.stale,1); self.assertFalse(b.publish(Message(0,1,'WAIT')))
    def test_concurrency(self):
        b=Blackboard(); done=threading.Event(); seen=[]
        def writer():
            for i in range(20000):
                b.publish(Message(i,0,str(i)))
                if i%16==0: time.sleep(0)
            done.set()
        t=threading.Thread(target=writer); t.start(); i=0; old=None
        while not done.is_set() or b.snapshot()[1] is not None:
            safe=i%4==0; m=b.promote(safe,0)
            if not safe: self.assertEqual(m,old)
            if m!=old:
                self.assertGreater(m.seq,-1 if old is None else old.seq); self.assertEqual(m.value,str(m.seq)); seen.append(m.seq)
            old=m;i+=1
            if i%16==0: time.sleep(0)
        t.join(); self.assertEqual(old.seq,19999); self.assertEqual(b.commits+b.overwritten,20000)
        print('Concorrência: 20.000 mensagens, commits=',b.commits,'substituídas=',b.overwritten,'violações=0')
    def test_prompt_scope(self):
        for sc in SCENARIOS:
            o=observe(World(sc,7),True); text=system_prompt(o,True)
            self.assertIn('Primary objective',text); self.assertIn('Every action including WAIT',text)
            self.assertEqual('NEGOTIATE' in choices_for(o,True),sc=='C_persona')
            self.assertEqual('offers' in o,sc=='C_persona')
    def test_logical_latency_independence(self):
        class Control:
            model='mock'; digest='MOCK-NOT-LLM'
            def __init__(self,delay): self.delay=delay
            def infer(self,o,choices,seed,previous):
                time.sleep(self.delay); goal=planner_goal(o); a=goal if o['strategic_mode'] else body(o,goal); a=a if a in choices else 'WAIT'
                return dict(action=a,note='mock',raw='mock',ms=self.delay*1000,tokens_in=0,tokens_out=0,parse_error=False)
        for sc in SCENARIOS:
            for mode in ('llm_puro','llm_hibrido'):
                a=run_episode(sc,20261119,mode,Control(0)); b=run_episode(sc,20261119,mode,Control(.003))
                self.assertEqual([x['executed'] for x in a['rows']],[x['executed'] for x in b['rows']]); self.assertEqual(a['summary']['success'],b['summary']['success'])
                self.assertEqual(b['summary']['inference_waits'],0); self.assertGreater(b['summary']['used_responses'],0)
                self.assertTrue(all(x['received_turn'] is None or x['received_turn']>=x['request_turn'] for x in b['calls']))
    def test_wait_voluntary(self):
        class Waiter:
            model='mock'; digest='MOCK-NOT-LLM'
            def infer(self,*args): return dict(action='WAIT',note='mock',raw='mock',ms=0,tokens_in=0,tokens_out=0,parse_error=False)
        for mode in ('llm_puro','llm_hibrido'):
            r=run_episode('B_obstaculos',7,mode,Waiter()); s=r['summary']; self.assertEqual(s['voluntary_waits'],s['deadline']); self.assertEqual(s['inference_waits'],0); self.assertEqual(s['success'],0); self.assertEqual(s['longest_wait_streak'],s['deadline'])
    def test_context_and_map(self):
        from experimento import goal_status
        for sc in SCENARIOS:
            w=World(sc,20261119)
            for event in (False,True):
                if event: w.t=w.event_time;w.trigger()
                o=observe(w,True); graph={tuple(sorted((a,x['to']))):x['energy_cost'] for a,neighbors in o['adjacency'].items() for x in neighbors}
                self.assertEqual(graph,{e:w.costs[e] for e in w.edges-w.closed})
                self.assertEqual(o['valid_actions'],w.actions());self.assertEqual(o['mission_status']['cargo_collected'],w.cargo)
                self.assertEqual(o['mission_status']['event_occurred'],w.event)
                self.assertNotIn('DELIVER',choices_for(o,True));self.assertIn('COLLECT',choices_for(o,True))
            w.cargo=True;w.energy=18;w.accepted=True;o=observe(w,True)
            self.assertNotIn('COLLECT',choices_for(o,True));self.assertNotIn('CHARGE',choices_for(o,True));self.assertNotIn('NEGOTIATE',choices_for(o,True));self.assertIn('DELIVER',choices_for(o,True));self.assertIn('WAIT',choices_for(o,True))
    def test_actual_overlap(self):
        from diagnostico_60hz import diagnostic
        class Control:
            model='mock';digest='MOCK-NOT-LLM'
            def infer(self,o,choices,seed,previous):
                time.sleep(.14);return dict(action='WAIT',note='mock',raw='mock',ms=140,tokens_in=0,tokens_out=0,parse_error=False)
        r=diagnostic(Control(),20261006)
        self.assertGreater(r['summary']['actions_during_inference'],0)
        self.assertTrue(all(not f['commit'] or f['safe'] for f in r['frames']))
    def test_http_async(self):
        s=ThreadingHTTPServer(('127.0.0.1',0),Handler); t=threading.Thread(target=s.serve_forever,daemon=True); t.start()
        try:
            c=Ollama('http://127.0.0.1:'+str(s.server_port),'mock:only'); self.assertEqual(c.preflight()['digest'],'MOCK-NOT-LLM')
            for m in ('llm_puro','llm_hibrido'):
                r=run_episode('B_obstaculos',20261119,m,c,tick_s=.1,period=3,frame_hz=60)
                self.assertGreater(r['summary']['calls'],0); self.assertTrue(all(not f['commit'] or f['safe'] for f in r['frames'])); self.assertGreater(len(r['frames']),0); self.assertTrue(all(c['ms']>=30 for c in r['calls'])); print('HTTP MOCK',m,'frames=',len(r['frames']),'misses=',r['summary']['missed_frames'],'calls=',r['summary']['calls'],'success=',r['summary']['success'])
        finally: s.shutdown();s.server_close();t.join()
if __name__=='__main__': unittest.main(verbosity=2)
