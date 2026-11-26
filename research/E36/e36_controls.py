#!/usr/bin/env python3
"""Open N39 causal history-swap and object-equivariance controls, no new oracle."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from e36_meta_ridge import Model,pack,evaluate
from e35_sequential import prep_node30
ROOT=Path(__file__).resolve().parent
rows=json.load(open(ROOT/'E34'/'REF'/'TWM_N39'/'DATA'/'N39_HELD_RECORDS.json'))['rows']
assert len(rows)==160
for j in range(0,160,2): assert rows[j]['pair_id']==rows[j+1]['pair_id']
swap=copy.deepcopy(rows)
for j in range(0,160,2):
    swap[j]['history'],swap[j+1]['history']=swap[j+1]['history'],swap[j]['history']
perm=np.array([2,0,3,1]); inv=np.argsort(perm)
def pn(x):return np.asarray(x,dtype='i1')[perm].tolist()
def pe(x):return np.asarray(x,dtype='i1').reshape(4,4)[np.ix_(perm,perm)].reshape(-1).tolist()
def remap_actor(a):return int(np.flatnonzero(perm==a)[0])
permrows=copy.deepcopy(rows)
for r in permrows:
    r['query_nodes']=pn(r['query_nodes']);r['query_edges']=pe(r['query_edges']);r['future_nodes']=pn(r['future_nodes']);r['future_edges']=pe(r['future_edges']);r['future_actions']=[remap_actor(a) for a in r['future_actions']]
    for t in r['history']:
        for k in ('nodes_before','nodes_after'): t[k]=pn(t[k])
        for k in ('edges_before','edges_after'):t[k]=pe(t[k])
        t['action']=remap_actor(t['action'])
out={}
for seed in (17,29,43):
    model=Model();ck=torch.load(ROOT/'E36_RESULTS'/f'E36_seed{seed}.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);model.eval()
    pred={}
    for name,rr in [('original',rows),('history_swap',swap),('permuted',permrows)]:
        op=pack(rr);packed=prep_node30(rr,op,model.edge)
        scores,p=evaluate(model,packed,batch_size=41)
        pred[name]=p
        out.setdefault(str(seed),{})[name]={'full_exact':scores['full_exact'],'edge_exact':scores['edge_exact'],'node_exact':scores['node_exact'],'exact_pairs':scores['exact_pairs']}
    p=pred['permuted'];o=pred['original']
    good_edge=np.array([np.array(m,dtype='i1').reshape(4,4)[np.ix_(perm,perm)][~np.eye(4,dtype=bool)] for m in [np.pad(row,(0,4)) for row in []]]) if False else None
    # Original pred edge is off-diagonal 12-bit; reconstruct complete mask before permutation.
    equiv=0
    for e1,e2,n1,n2 in zip(o['pe'],p['pe'],o['pn'],p['pn']):
        full=np.zeros((4,4),dtype='i1');full[~np.eye(4,dtype=bool)]=e1
        equiv+=int(np.array_equal(full[np.ix_(perm,perm)][~np.eye(4,dtype=bool)],e2) and np.array_equal(n1[perm],n2))
    out[str(seed)]['equivariance_full_pred_n_of_160']=equiv
    print('CTRL',seed,out[str(seed)],flush=True)
(ROOT/'E36_RESULTS'/'E36_CAUSAL_CONTROLS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
