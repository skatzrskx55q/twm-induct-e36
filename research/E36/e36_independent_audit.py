#!/usr/bin/env python3
"""Checks E36 using separately recomputed truth for all open sets and N2 traces.
Do not pass future/label data to model prediction computations."""
from pathlib import Path
import sys, json,hashlib,random
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
from e36_meta_ridge import Model,pack,tensors,evaluate,train_solver,seeds
from e35_sequential import prep_node30
from e35_n2_rollout import HORIZONS,SPLITS
OUT=ROOT/'E36_RESULTS'
torch.set_num_threads(4)
rows={k:json.load(open(ROOT/'E34'/'INPUTS'/(k+'_records.json')))['rows'] for k in ['train','known_val','paired_known']}
rows['N39_open']=json.load(open(ROOT/'E34'/'REF'/'TWM_N39'/'DATA'/'N39_HELD_RECORDS.json'))['rows']
orig={k:pack(v) for k,v in rows.items()}
allmetr={};allraw={};max_gap=0;checks=0
for seed in (17,29,43):
    ck=torch.load(OUT/f'E36_seed{seed}.pt',map_location='cpu',weights_only=False)
    assert ck['seed']==seed and ck['steps_edge']==750 and ck['steps_node']==750
    model=Model();model.load_state_dict(ck['model']);model.eval()
    packs={k:prep_node30(rows[k],orig[k],model.edge) for k in rows}
    allmetr[str(seed)]={'train_log_edge':ck['log_edge'],'train_log_node':ck['log_node'],'splits':{}}
    for k in ('train','known_val','paired_known','N39_open'):
        met,pred=evaluate(model,packs[k],batch_size=37)
        allmetr[str(seed)]['splits'][k]=met
        for field,val in pred.items(): allraw[f'{seed}__{k}__{field}']=val
        # independent truth labels directly from observed record fields (not predicted outputs).
        ye=[];yn=[]
        for r in rows[k]:
            v=np.array(r['query_edges'],dtype=np.int8).reshape(4,4)^np.array(r['future_edges'],dtype=np.int8).reshape(4,4)
            ye.append(v[~np.eye(4,dtype=bool)]);yn.append(np.array(r['query_nodes'],dtype=np.int8)^np.array(r['future_nodes'],dtype=np.int8))
        ye=np.asarray(ye);yn=np.asarray(yn)
        assert np.array_equal(ye,pred['ye']) and np.array_equal(yn,pred['yn'])
        good_e=(pred['pe']==ye).all(axis=1);good_n=(pred['pn']==yn).all(axis=1)
        assert int((good_e&good_n).sum())==met['full_exact']
        assert met['false_edge_flip']==int((pred['pe']&~ye.astype(bool)).sum())
        assert met['missed_edge_flip']==int((~pred['pe'].astype(bool)&ye.astype(bool)).sum())
        if met['exact_pairs'] is not None:
            assert met['exact_pairs']==sum(bool((good_e&good_n)[i] and (good_e&good_n)[i+1]) for i in range(0,len(ye),2))
        checks+=len(rows[k])*(12+4)
print('OPEN CHECKS',checks,flush=True)
(OUT/'E36_METRICS.json').write_text(json.dumps(allmetr,ensure_ascii=False,indent=2))
np.savez_compressed(OUT/'E36_RAW.npz',**allraw)
# independent N2 reference-matched trajectory evaluation (not using predictor code)
folder=ROOT/'E34'/'REF'/'TWM_N2'/'N2_DATA'
X=[json.loads(line) for line in open(folder/'N2_MODEL_INPUTS.jsonl')]
Y={r['episode_id']:r for r in (json.loads(line) for line in open(folder/'N2_PRIVATE_LABELS.jsonl'))}
assert len(X)==len(Y)==2368
scores={}
for seed in (17,29,43):
    agg={split:{str(h):{'n':0,'exact':0,'path_exact':0,'edge_exact':0,'node_exact':0} for h in HORIZONS} for split in SPLITS}
    raw=[json.loads(l) for l in open(OUT/f'E36_N2_RAW_seed{seed}.jsonl')]
    assert len(raw)==sum(1 for x in X if x['split'] in SPLITS)
    for pred,x in zip(raw,[x for x in X if x['split'] in SPLITS]):
        assert pred['episode_id']==x['episode_id'] and pred['split']==x['split']
        y=Y[x['episode_id']];assert len(pred['trace'])==16
        path=True
        for t,(state,true_n,true_e) in enumerate(zip(pred['trace'],y['future_nodes'],y['future_edges']),1):
            n_ok=state[0]==true_n;e_ok=state[1]==true_e;path&=n_ok and e_ok
            if t in HORIZONS:
                m=agg[x['split']][str(t)];m['n']+=1;m['node_exact']+=int(n_ok);m['edge_exact']+=int(e_ok);m['exact']+=int(n_ok and e_ok);m['path_exact']+=int(path)
            checks+=20
    scores[str(seed)]=agg
    print('N2AUDIT',seed,'L3',[(h,agg['test_L3'][str(h)]['exact']) for h in HORIZONS],'L4',[(h,agg['test_L4'][str(h)]['exact']) for h in HORIZONS],flush=True)
(OUT/'E36_N2_ROLLOUT.json').write_text(json.dumps({'source':'archived open N2','horizons':HORIZONS,'seeds':scores},indent=2))
# full repeat of seed17 training, check weights and optimizer tensor values
print('RETRAIN17',flush=True)
seeds(17);model=Model();optedge,_=train_solver(model,orig['train'],'edge',17,steps=750,batch_size=16)
packs={k:prep_node30(rows[k],orig[k],model.edge) for k in ['train']}
optnode,_=train_solver(model,packs['train'],'node',17,steps=750,batch_size=16)
old=torch.load(OUT/'E36_seed17.pt',map_location='cpu',weights_only=False)
max_gap=max(float((v-old['model'][k]).abs().max()) for k,v in model.state_dict().items())
assert max_gap==0.0
for new,stored in [(optedge,old['optimizer_edge']),(optnode,old['optimizer_node'])]:
    assert new.keys()==stored.keys()
    for p,st in new['state'].items():
        for k,v in st.items():
            sv=stored['state'][p][k]
            if torch.is_tensor(v): assert torch.equal(v,sv),(p,k)
            else:assert v==sv,(p,k)
report={'status':'PASS','checkpoint_seed_count':3,'independent_checks_approx':checks,'n2_episodes':len(X),'all_N2_traces_per_seed':len(raw),'direct_truth_reconstruction':True,'N2_recomputed_from_raw':True,'repeat_train_seed17_bitwise_identical':True,'max_weights_diff_retrain_seed17':max_gap}
(OUT/'E36_AUDIT.json').write_text(json.dumps(report,indent=2));print('PASS',report,flush=True)
