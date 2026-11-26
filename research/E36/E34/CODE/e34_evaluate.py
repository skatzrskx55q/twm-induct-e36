#!/usr/bin/env python3
"""Evaluation code only; reads future targets exclusively in `score`."""
import argparse,hashlib,json,time,sys
from pathlib import Path
import numpy as np
from e34_induction import infer
ROOT=Path(__file__).resolve().parents[1]
SEEDS=(17,29,43)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def score(rows,seed):
    metrics={'n':len(rows),'full_exact':0,'edge_exact':0,'node_exact':0,'false_edge_flip':0,'missed_edge_flip':0,'false_node_flip':0,'missed_node_flip':0,'per_law':{}}
    preds_e=[];preds_n=[];target_e=[];target_n=[];pred_exact=[]
    for i,row in enumerate(rows):
        # The ONLY information entering infer: observed transitions, current graph, and action.
        pe,pn=infer(row['history'],row['query_nodes'],row['query_edges'],row['future_actions'][0],seed)
        n=row['n_nodes'];e0=np.asarray(row['query_edges'],dtype=np.int8).reshape(n,n);ef=np.asarray(row['future_edges'],dtype=np.int8).reshape(n,n)
        y_e=e0^ef;y_n=np.asarray(row['query_nodes'],dtype=np.int8)^np.asarray(row['future_nodes'],dtype=np.int8)
        valid=~np.eye(n,dtype=bool)
        ok_e=bool(np.array_equal(pe[valid],y_e[valid]));ok_n=bool(np.array_equal(pn,y_n));ok=ok_e and ok_n
        metrics['full_exact']+=int(ok);metrics['edge_exact']+=int(ok_e);metrics['node_exact']+=int(ok_n)
        metrics['false_edge_flip']+=int(np.sum((pe==1)&(y_e==0)&valid));metrics['missed_edge_flip']+=int(np.sum((pe==0)&(y_e==1)&valid))
        metrics['false_node_flip']+=int(np.sum((pn==1)&(y_n==0)));metrics['missed_node_flip']+=int(np.sum((pn==0)&(y_n==1)))
        law=row.get('law_id','UNKNOWN');sub=metrics['per_law'].setdefault(law,{'n':0,'full':0,'edge':0,'node':0,'false_edge':0})
        sub['n']+=1;sub['full']+=int(ok);sub['edge']+=int(ok_e);sub['node']+=int(ok_n);sub['false_edge']+=int(np.sum((pe==1)&(y_e==0)&valid))
        preds_e.append(pe.copy());preds_n.append(pn.copy());target_e.append(y_e.copy());target_n.append(y_n.copy());pred_exact.append(int(ok))
    if len(rows)%2==0 and all(r.get('pair_id') and r.get('pair_id')==rows[j^1].get('pair_id') for j,r in enumerate(rows)):
        metrics['exact_pairs']=sum(int(pred_exact[j] and pred_exact[j+1]) for j in range(0,len(rows),2))
    arrays={name:np.asarray(v,dtype=np.int8) for name,v in [('pred_edge_flip',preds_e),('pred_node_flip',preds_n),('true_edge_flip',target_e),('true_node_flip',target_n),('full_exact',pred_exact)]}
    return metrics,arrays

def main():
    p=json.load(open(ROOT/'E34_FROZEN_PROTOCOL.json'));assert p['status']=='FROZEN_BEFORE_EXTERNAL_EVALUATION'
    for f,expected in p['frozen_sha256'].items():assert sha(ROOT/f)==expected,f
    started=time.time();res={'stage':'E34','method':'within_episode_CART_6_6','seeds':{},'frozen_protocol_sha256':sha(ROOT/'E34_FROZEN_PROTOCOL.json'),'source':'E33 train/val and N39 already opened'}
    paths={'known_val':ROOT/'INPUTS/known_val_records.json','paired_known':ROOT/'INPUTS/paired_known_records.json','N39_open':ROOT/'REF/TWM_N39/DATA/N39_HELD_RECORDS.json'}
    raw={}
    for seed in SEEDS:
        res['seeds'][str(seed)]={}
        for name,path in paths.items():
            rows=json.load(open(path))['rows'];m,a=score(rows,seed);res['seeds'][str(seed)][name]=m
            for k,v in a.items(): raw[f'seed{seed}__{name}__{k}']=v
            print('SCORE',seed,name,m['full_exact'],'/',m['n'],'pairs',m.get('exact_pairs'),'edge',m['edge_exact'],'node',m['node_exact'],flush=True)
    res['seconds']=round(time.time()-started,3)
    (ROOT/'RESULTS/E34_METRICS.json').write_text(json.dumps(res,ensure_ascii=False,indent=2))
    np.savez_compressed(ROOT/'RESULTS/E34_RAW_PREDICTIONS.npz',**raw)
    print('TOTAL_SECONDS',res['seconds'],flush=True)
if __name__=='__main__':main()
