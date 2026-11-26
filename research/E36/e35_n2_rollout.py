#!/usr/bin/env python3
"""E35 pure neural memory evaluation on archived N2; evaluator labels never enter model predictor."""
import os,sys,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from e35_sequential import SequentialNet
sys.path.insert(0,str(ROOT/'E34'/'CODE'))
from e34_induction import local_features,node_features
SPLITS=('ID_state','OOD_structure','test_L2','test_L3','test_L4','ambiguous'); HORIZONS=(1,4,8,16)

def make_predictor(model,history,n):
    good=~np.eye(n,dtype=bool)
    es=[];ey=[];ns=[];ny=[]
    for tr in history[-8:]:
        e,_=local_features(tr['nodes_before'],tr['edges_before'],tr['action'])
        es.append(e[good]);ey.append((np.asarray(tr['edges_before'],dtype='i1').reshape(n,n)^np.asarray(tr['edges_after'],dtype='i1').reshape(n,n))[good]);
        ns.append(node_features(tr['nodes_before'],tr['edges_before'],tr['edges_after'],tr['action']));ny.append(np.asarray(tr['nodes_before'],dtype='i1')^np.asarray(tr['nodes_after'],dtype='i1'))
    es=torch.as_tensor(np.concatenate(es),dtype=torch.float32)[None];ey=torch.as_tensor(np.concatenate(ey),dtype=torch.float32)[None];
    ns=torch.as_tensor(np.concatenate(ns),dtype=torch.float32)[None];ny=torch.as_tensor(np.concatenate(ny),dtype=torch.float32)[None]
    # feature representations cache for 16 steps; no labels from future are used
    with torch.no_grad():
        ek=torch.nn.functional.normalize(model.edge.net(es),dim=-1)
        nk=torch.nn.functional.normalize(model.node.net(ns),dim=-1)
        e_temperature=model.edge.log_temperature.exp()*5;n_temperature=model.node.log_temperature.exp()*5
    def attention(q,k,y,temp,net):
        z=torch.nn.functional.normalize(net(torch.as_tensor(q,dtype=torch.float32)[None]),dim=-1)
        w=torch.softmax((z@k.transpose(1,2))*temp,dim=-1)
        return (w@y[:,:,None]).squeeze().numpy()
    def pred(nodes,edges,actor):
        e,_=local_features(nodes,edges,actor)
        with torch.no_grad():
            em=attention(e[good],ek,ey,e_temperature,model.edge.net)>=.5
            ep=np.asarray(edges,dtype='i1').reshape(n,n).copy();ep[good]^=em.astype('i1')
            qnode=node_features(nodes,edges,ep,actor)
            nm=attention(qnode,nk,ny,n_temperature,model.node.net)>=.5
        return np.asarray(nodes,dtype='i1')^nm.astype('i1'),ep.reshape(-1)
    return pred

def main():
    torch.set_num_threads(4);p=ROOT/'E34'/'REF'/'TWM_N2'/'N2_DATA';xs=[json.loads(l) for l in open(p/'N2_MODEL_INPUTS.jsonl')]; ys=[json.loads(l) for l in open(p/'N2_PRIVATE_LABELS.jsonl')]; assert len(xs)==len(ys)==2368
    out={'frozen_historical_N2':True,'splits':{},'seeds':{},'horizons':HORIZONS};t=time.time()
    for seed in [17,29,43]:
        z=torch.load(ROOT/'RESULTS'/f'E35_sequential_seed{seed}.pt',map_location='cpu',weights_only=False)
        model=SequentialNet();model.load_state_dict(z['model']);model.eval();raw_rows=[];scores={split:{str(h):{'n':0,'exact':0,'path_exact':0,'edge_exact':0,'node_exact':0} for h in HORIZONS} for split in SPLITS}
        for i,(x,y) in enumerate(zip(xs,ys)):
            if x['split'] not in SPLITS:continue
            assert x['episode_id']==y['episode_id']
            split=x['split'];n=x['n_nodes'];predict=make_predictor(model,x['history'],n)
            nodes=np.asarray(x['query_nodes'],dtype='i1');edges=np.asarray(x['query_edges'],dtype='i1');pathok=True;pred_trace=[]
            for step,(actor,yn,ye) in enumerate(zip(x['future_actions'],y['future_nodes'],y['future_edges']),1):
                nodes,edges=predict(nodes,edges,actor)
                pred_trace.append([nodes.tolist(),edges.tolist()])
                ngood=np.array_equal(nodes,yn);egood=np.array_equal(edges,ye);pathok &= ngood and egood
                if step in HORIZONS:
                    item=scores[split][str(step)];item['n']+=1;item['exact']+=int(ngood and egood);item['path_exact']+=int(pathok);item['node_exact']+=int(ngood);item['edge_exact']+=int(egood)
            raw_rows.append({'episode_id':x['episode_id'],'split':split,'seed':seed,'trace':pred_trace})
        with open(ROOT/'RESULTS'/f'E35_N2_RAW_seed{seed}.jsonl','w') as f:
            for line in raw_rows:f.write(json.dumps(line,separators=(',',':'))+'\n')
        out['seeds'][str(seed)]=scores
        print('N2',seed,'elapsed',round(time.time()-t,1),'L3',[(h,scores['test_L3'][str(h)]['exact'],scores['test_L3'][str(h)]['path_exact']) for h in HORIZONS],'L4',[(h,scores['test_L4'][str(h)]['exact']) for h in HORIZONS],flush=True)
    (ROOT/'RESULTS'/'E35_N2_ROLLOUT.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
