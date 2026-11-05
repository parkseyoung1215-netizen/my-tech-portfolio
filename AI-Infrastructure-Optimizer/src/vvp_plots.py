"""Standalone PNG/SVG figures from saved measurements; no edited performance data."""
import csv,hashlib,json,os,sys
from pathlib import Path

def plots(prefix):
    prefix=Path(prefix).resolve();deps=Path(os.environ.get('VVP_PLOT_DEPS','/private/tmp/vvp_plot_deps_20261105'))
    if deps.exists():sys.path.insert(0,str(deps))
    os.environ.setdefault('MPLCONFIGDIR','/private/tmp/vvp_matplotlib_config')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def read(name):
        with Path(f'{prefix}_{name}.csv').open() as f:return list(csv.DictReader(f))
    rows=read('aggregate');models=list(dict.fromkeys(r['model'] for r in rows));methods=['none','full','selective','rcav','improved','vvp']
    labels={'none':'No verification','full':'Full / uniform','selective':'Selective','rcav':'RCAV','improved':'Improved RCAV','vvp':'VVP'}
    colors={'none':'#666666','full':'#d17c18','selective':'#1675b8','rcav':'#188850','improved':'#a35424','vvp':'#8c48b0'}
    targets=[Path(f'{prefix}_{name}.{ext}') for name in ('accuracy_budget','calls_budget','accuracy_cpu','accuracy_accounted_cpu','transitions') for ext in ('png','svg')]
    if any(p.exists() for p in targets):raise FileExistsError('Do not overwrite figures')
    def save(fig,name):
        for ext in ('png','svg'):
            with Path(f'{prefix}_{name}.{ext}').open('xb') as f:fig.savefig(f,format=ext,dpi=180,bbox_inches='tight')
        plt.close(fig)
    for metric,name,ylabel,scale in [('accuracy','accuracy_budget','Final valid-answer accuracy (%)',100),('verification_calls','calls_budget','Actual verifier calls',1)]:
        fig,axes=plt.subplots(1,len(models),figsize=(12,4),squeeze=False)
        for ax,m in zip(axes[0],models):
            for method in methods:
                rr=sorted([r for r in rows if r['model']==m and r['method']==method],key=lambda r:int(r['budget_percent']))
                x=[int(r['budget_percent']) for r in rr];y=[float(r[metric+'_mean'])*scale for r in rr]
                lo=[float(r[metric+'_ci95_low'])*scale for r in rr];hi=[float(r[metric+'_ci95_high'])*scale for r in rr]
                ax.plot(x,y,marker='s' if method=='vvp' else 'o',label=labels[method],color=colors[method],linestyle='--' if method=='none' else '-')
                ax.fill_between(x,lo,hi,color=colors[method],alpha=.12)
            ax.set_title(m);ax.set_xlabel('Nominal verifier call cap (%)');ax.set_ylabel(ylabel);ax.grid(alpha=.25);ax.set_xticks([0,10,25,50,75,100])
        axes[0][-1].legend(fontsize=8,loc='best');fig.suptitle('Frozen initial answers; 5 actual repeats; bands = repeat-mean bootstrap CI')
        fig.tight_layout();save(fig,name)
    accounting=read('accounting');ai={(r['model'],r['method'],r['budget_percent']):r for r in accounting}
    for accounted,name in [(False,'accuracy_cpu'),(True,'accuracy_accounted_cpu')]:
        fig,axes=plt.subplots(1,len(models),figsize=(12,4),squeeze=False)
        for ax,m in zip(axes[0],models):
            for method in methods:
                rr=sorted([r for r in rows if r['model']==m and r['method']==method],key=lambda r:int(r['budget_percent']))
                x=[float(ai[m,method,r['budget_percent']]['accounted_full_cpu']) if accounted else float(r['measured_cpu_seconds_mean']) for r in rr]
                y=[float(r['accuracy_mean'])*100 for r in rr]
                ax.plot(x,y,'o-',color=colors[method],label=labels[method],alpha=.85)
                for xx,yy,r in zip(x,y,rr):
                    if method=='vvp' and r['budget_percent']=='25':ax.annotate('VVP 25%',(xx,yy),xytext=(10,-22),textcoords='offset points',arrowprops={'arrowstyle':'->','color':colors[method]},fontsize=8)
            ax.set_title(m);ax.set_ylabel('Final valid-answer accuracy (%)');ax.set_xlabel('Accounted initial + measured stage CPU (s)' if accounted else 'Measured planning + verification CPU (s)');ax.grid(alpha=.25)
        axes[0][-1].legend(fontsize=8);fig.suptitle('Component accounting, not independent production execution' if accounted else 'Observed accuracy / CPU trade-off; no posthoc winner selection')
        fig.tight_layout();save(fig,name)
    fig,axes=plt.subplots(1,len(models),figsize=(12,4),squeeze=False)
    selected=[('none',0),('full',100),('selective',100),('rcav',100),('improved',100),('vvp',25)]
    for ax,m in zip(axes[0],models):
        rr=[next(r for r in rows if (r['model'],r['method'],int(r['budget_percent']))==(m,method,b)) for method,b in selected]
        x=list(range(len(rr)));ax.bar(x,[float(r['repairs_mean']) for r in rr],color='#2a9b55',label='Repairs')
        ax.bar(x,[-float(r['damage_mean']) for r in rr],color='#c74c45',label='Damage')
        ax.scatter(x,[float(r['missed_repairs_full_reference_mean']) for r in rr],color='#d58d12',marker='D',label='Missed: Full100 reference')
        ax.set_xticks(x,[labels[method]+' '+str(b)+'%' for method,b in selected],rotation=30,ha='right');ax.set_title(m);ax.set_ylabel('Mean changed / missed answers per 300');ax.axhline(0,color='black',linewidth=.7);ax.grid(axis='y',alpha=.25)
    axes[0][-1].legend(fontsize=8);fig.suptitle('Matched Full100 reference ran separately; not exact same-time counterfactual')
    fig.tight_layout();save(fig,'transitions')
    provenance={'matplotlib_version':matplotlib.__version__,'source_sha256':{s:hashlib.sha256(Path(f'{prefix}_{s}.csv').read_bytes()).hexdigest() for s in ('aggregate','accounting')},
        'figures':[str(p.name) for p in targets],'note':'No smoothed, fabricated or selected performance values. Coincident curves may overlap.'}
    with Path(f'{prefix}_plot_metadata.json').open('x') as f:json.dump(provenance,f,indent=2)
    print(json.dumps(provenance,indent=2))

if __name__=='__main__':plots(sys.argv[1])
