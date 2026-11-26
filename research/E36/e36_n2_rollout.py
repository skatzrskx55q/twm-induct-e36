#!/usr/bin/env python3
"""E36 N2 fully autonomous rollout; labels read only for scoring, never by predictor."""
from pathlib import Path
import json,sys,time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
from e36_meta_ridge import Model,seeds
from e35_n2_rollout import SPLITS,HORIZONS
sys.path.insert(0,str(ROOT/'E34'/'CODE'))
from e34_induction import local_features,node_features

def make_predictor(model,history,n):
    non_diag=~np.eye(n,dtype=bool)
    ex=[];ey=[];nx=[];ny=[]
    for tr in history[-8:]:
        e,_=local_features(tr['nodes_before'],tr['edges_before'],tr['action'])
        ex.append(e[non_diag]);ey.append((np.asarray(tr['edges_before'],dtype='i1').reshape(n,n)^np.asarray(tr['edges_after'],dtype='i1').reshape(n,n))[non_diag])
        nx.append(node_features(tr['nodes_before'],tr['edges_before'],tr['edges_after'],tr['action']))
        ny.append(np.asarray(tr['nodes_before'],dtype='i1')^np.asarray(tr['nodes_after'],dtype='i1'))
    xs=[torch.from_numpy(np.concatenate(ex))[None],torch.from_numpy(np.concatenate(nx))[None]]
    ys=[torch.as_tensor(np.concatenate(ey),dtype=torch.float32)[None],torch.as_tensor(np.concatenate(ny),dtype=torch.float32)[None]]
    with torch.no_grad():
        weights=[]
        for head,x,y in zip([model.edge,model.node],xs,ys):
            X=head._embed(x);xt=X.transpose(1,2)
            d=X.shape[-1];la=torch.nn.functional.softplus(head.log_lambda)+0.02
            w=torch.linalg.solve(xt@X+la*torch.eye(d)[None],xt@y[...,None]);weights.append(w)
        def predict_one(head, features, w):
            q=torch.from_numpy(np.asarray(features,dtype='f4'))[None]
            raw=(head._embed(q)@w).squeeze(-1)
            return ((raw>=.5).squeeze(0)).numpy().astype('i1')
    def pred(nodes,edges,actor):
        with torch.no_grad():
            e,_=local_features(nodes,edges,actor)
            em=predict_one(model.edge,e[non_diag],weights[0]);ep=np.asarray(edges,dtype='i1').reshape(n,n).copy()
            ep[non_diag]^=em
            u=node_features(nodes,edges,ep,actor)
            nm=predict_one(model.node,u,weights[1])
        return np.asarray(nodes,dtype='i1')^nm,ep.reshape(-1)
    return pred

def main():
    torch.set_num_threads(4); folder=ROOT/'E34'/'REF'/'TWM_N2'/'N2_DATA';
    xs=[json.loads(l) for l in open(folder/'N2_MODEL_INPUTS.jsonl')]
    ys=[json.loads(l) for l in open(folder/'N2_PRIVATE_LABELS.jsonl')]
    assert len(xs)==len(ys)==2368
    out={'source':'N2 frozen history - already open','horizons':HORIZONS,'seeds':{}}
    st=time.time()
    for seed in (17,29,43):
        ck=torch.load(ROOT/'E36_RESULTS'/f'E36_seed{seed}.pt',map_location='cpu',weights_only=False)
        model=Model();model.load_state_dict(ck['model']);model.eval()
        scores={split:{str(h):{'n':0,'exact':0,'path_exact':0,'node_exact':0,'edge_exact':0} for h in HORIZONS} for split in SPLITS}
        raw=[]
        for i,(x,y) in enumerate(zip(xs,ys)):
            if x['split'] not in SPLITS: continue
            assert x['episode_id']==y['episode_id']
            split=x['split'];predict=make_predictor(model,x['history'],x['n_nodes']);nodes=np.asarray(x['query_nodes'],dtype='i1');edges=np.asarray(x['query_edges'],dtype='i1')
            pathok=True;trace=[]
            for t,(actor,yn,ye) in enumerate(zip(x['future_actions'],y['future_nodes'],y['future_edges']),1):
                nodes,edges=predict(nodes,edges,actor)
                trace.append([nodes.tolist(),edges.tolist()]);ngood=np.array_equal(nodes,yn);egood=np.array_equal(edges,ye);pathok &=ngood and egood
                if t in HORIZONS:
                    point=scores[split][str(t)];point['n']+=1;point['exact']+=int(ngood and egood);point['path_exact']+=int(pathok);point['node_exact']+=int(ngood);point['edge_exact']+=int(egood)
            raw.append({'episode_id':x['episode_id'],'split':split,'seed':seed,'trace':trace})
            if i and i%600==0:print('PROGRESS',seed,i,round(time.time()-st,1),flush=True)
        out['seeds'][str(seed)]=scores
        with open(ROOT/'E36_RESULTS'/f'E36_N2_RAW_seed{seed}.jsonl','w') as f:
            for r in raw:f.write(json.dumps(r,separators=(',',':'))+'\n')
        print('N2',seed,'elapsed',round(time.time()-st,1),'L3',[(h,scores['test_L3'][str(h)]['exact'],scores['test_L3'][str(h)]['path_exact']) for h in HORIZONS], 'L4',[(h,scores['test_L4'][str(h)]['exact']) for h in HORIZONS],flush=True)
    (ROOT/'E36_RESULTS'/'E36_N2_ROLLOUT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
