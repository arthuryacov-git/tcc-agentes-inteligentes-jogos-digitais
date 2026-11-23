import csv,json,tkinter as tk
from tkinter import ttk,messagebox
from pathlib import Path
from simulador import POS,BASE_EDGES,edge

def launch(path):
    if not path.exists(): raise SystemExit(f'Arquivo inexistente: {path}')
    with path.open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    data={}
    for r in rows: data.setdefault((r['scenario'],r['mode'],r['seed']),[]).append(r)
    root=tk.Tk(); root.title('Laboratório de Agentes v5 — reprodução de episódios'); root.geometry('1000x650'); root.configure(bg='#f0f4fa')
    outer=ttk.Frame(root,padding=16); outer.pack(fill='both',expand=True)
    ttk.Label(outer,text='LABORATÓRIO DE AGENTES / V5',font=('Arial',19,'bold')).pack(anchor='w')
    ttk.Label(outer,text='Reprodução de dados medidos. V5: reprodução de decisões e objetivos registrados.').pack(anchor='w',pady=(0,12))
    bar=ttk.Frame(outer); bar.pack(fill='x'); variables=[tk.StringVar() for _ in range(3)]; boxes=[]
    for label,var,i in zip(['Cenário','Agente','Semente'],variables,range(3)):
        ttk.Label(bar,text=label).pack(side='left',padx=4); cb=ttk.Combobox(bar,textvariable=var,state='readonly',width=19); cb.pack(side='left',padx=4); boxes.append(cb)
    canvas=tk.Canvas(outer,height=340,bg='#142539',highlightthickness=0); canvas.pack(fill='both',expand=True,pady=10)
    info=tk.StringVar(); ttk.Label(outer,textvariable=info,wraplength=920,font=('Arial',11)).pack(anchor='w')
    buttons=ttk.Frame(outer); buttons.pack(fill='x',pady=10); i=0; playing=False; timer=None
    def current(): return data.get(tuple(x.get() for x in variables),[])
    def draw():
        rs=current(); canvas.delete('all')
        if not rs: return
        r=rs[i]; w=max(canvas.winfo_width(),750); h=max(canvas.winfo_height(),300); xy=lambda n:(65+POS[n][0]*(w-130)/5,55+POS[n][1]*(h-110)/2)
        closed={tuple(v) for v in json.loads(r['closed'])}
        for a,b in BASE_EDGES:
            canvas.create_line(*xy(a),*xy(b),width=3,fill='#f87171' if edge(a,b) in closed else '#617990',dash=(6,3) if edge(a,b) in closed else ())
        for n in POS:
            x,y=xy(n); c='#55cda2' if n==r['charger'] else '#eab95c' if n==r['pickup'] else '#b39ae9' if n==r['target'] else '#8ba0b5'; canvas.create_oval(x-19,y-19,x+19,y+19,fill=c,outline=''); canvas.create_text(x,y,text=n,fill='#142539',font=('Arial',13,'bold'))
        x,y=xy(r['location_after']); canvas.create_oval(x-25,y-25,x+25,y+25,outline='#4bd8ff',width=4)
        canvas.create_text(12,10,anchor='nw',text='Azul: agente  •  Verde: recarga  •  Amarelo: coleta  •  Roxo: destino  •  Vermelho: bloqueio',fill='white',font=('Arial',10))
        info.set(f"Passo {i+1}/{len(rs)} | Turno {r['turn']} | Energia {r['energy']} → {r['energy_after']} | Créditos {r['credits']} | Meta {r['goal']} | Ação {r['executed']} | Resultado {r['result']}\nEvento: {r['event'] or '—'} | Carga: {r['cargo']} | Acordo: {r['accepted']} | Intenção: {r['note'] or '—'}")
    def refresh(event=None):
        nonlocal i,playing
        playing=False; i=0
        for j,box in enumerate(boxes):
            choices=sorted({k[j] for k in data if all(k[z]==variables[z].get() for z in range(j))}); box['values']=choices
            if variables[j].get() not in choices: variables[j].set(choices[0] if choices else '')
        draw()
    def advance(step):
        nonlocal i
        i=max(0,min(len(current())-1,i+step)); draw()
    def play():
        nonlocal playing,timer
        if not playing: return
        if i>=len(current())-1: playing=False; return
        advance(1); timer=root.after(350,play)
    def toggle():
        nonlocal playing,timer
        playing=not playing
        if timer: root.after_cancel(timer); timer=None
        if playing: play()
    for box in boxes: box.bind('<<ComboboxSelected>>',refresh)
    for label,fn in [('Anterior',lambda:advance(-1)),('Próximo',lambda:advance(1)),('Reproduzir / pausar',toggle)]: ttk.Button(buttons,text=label,command=fn).pack(side='left',padx=4)
    canvas.bind('<Configure>',lambda _:draw()); refresh(); root.mainloop()

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("folder",nargs="?",default="resultados/v5_principal"); a=p.parse_args(); launch(Path(a.folder)/"decisoes.csv")
