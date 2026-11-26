#!/usr/bin/env python3
"""A new toy example: infer next state from eight observed transitions, no future label passed to model."""
from pathlib import Path
import sys,json,os
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD']='1'
import numpy as np
root=Path(__file__).resolve().parent/'research/E36'
# Toy history: the acting node flips, edges are unchanged. This is an example, not an accuracy claim.
n=4;edges=np.zeros((n,n),dtype=np.int8);nodes=np.array([0,1,0,1],dtype=np.int8);history=[]
for i in range(8):
    actor=i%n;after=nodes.copy();after[actor]^=1
    history.append(dict(action=actor,nodes_before=nodes.copy(),nodes_after=after.copy(),edges_before=edges.copy().reshape(-1),edges_after=edges.copy().reshape(-1)));nodes=after
actor=2
sys.path.insert(0,str(root))
import torch
from e36_meta_ridge import Model
from e36_n2_rollout import make_predictor
model=Model();ck=torch.load(root/'E36_RESULTS/E36_seed17.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model']);model.eval()
predict=make_predictor(model,history,n)
pred_nodes,pred_edges=predict(nodes,edges.reshape(-1),actor)
print(json.dumps(dict(support_count=len(history),current_nodes=nodes.tolist(),action=actor,predicted_nodes=pred_nodes.tolist(),predicted_edges=np.asarray(pred_edges).reshape(n,n).tolist()),indent=2))
