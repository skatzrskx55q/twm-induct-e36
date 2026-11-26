"""Exact N7ObjectDynamics architectural implementation retained for N16; no simulator imported."""
import torch
from torch import nn
ST,NE,NN,MAXN,HB = 46,12,10,12,8
class N7ObjectDynamics(nn.Module):
 def __init__(self):
  super().__init__()
  self.infer=nn.Sequential(nn.Linear(ST,48),nn.ReLU(),nn.Linear(48,32),nn.Tanh())
  self.edge_embed=nn.Sequential(nn.Linear(NE+HB+32,32),nn.ReLU())
  self.node_embed=nn.Sequential(nn.Linear(NN+HB+32,32),nn.ReLU())
  self.message=nn.Sequential(nn.Linear(96,32),nn.ReLU())
  self.node_refine=nn.Sequential(nn.Linear(96,32),nn.ReLU())
  self.edge_out=nn.Sequential(nn.Linear(96,32),nn.ReLU(),nn.Linear(32,1))
  self.node_out=nn.Sequential(nn.Linear(32+4+32,32),nn.ReLU(),nn.Linear(32,1))
 def forward(self,h,xe,xn,me,mn,he,hn):
  B=xe.size(0);n=xe.size(1);assert n==MAXN
  z=self.infer(h)
  edge=self.edge_embed(torch.cat((xe,he,z[:,None,None,:].expand(-1,n,n,-1)),dim=-1))
  node=self.node_embed(torch.cat((xn,hn,z[:,None,:].expand(-1,n,-1)),dim=-1))
  source=node[:,:,None,:].expand(-1,-1,n,-1);target=node[:,None,:,:].expand(-1,n,-1,-1)
  message=self.message(torch.cat((edge,source,target),dim=-1))*me[...,None]
  denom=mn.sum(1).clamp_min(2)[:,None,None]
  inc=message.sum(1)/denom;out=message.sum(2)/denom
  refined=self.node_refine(torch.cat((node,inc,out),dim=-1))
  ei=refined[:,:,None,:].expand(-1,-1,n,-1);ej=refined[:,None,:,:].expand(-1,n,-1,-1)
  e_logits=self.edge_out(torch.cat((edge,ei,ej),dim=-1)).squeeze(-1)
  # differentiable predicted adjacency, no rule execution
  cur=xe[...,0];flip=torch.sigmoid(e_logits)
  after=(cur*(1-flip)+(1-cur)*flip)*me
  onbits=xn[...,0]*mn;nden=(mn.sum(1)-1).clamp_min(1)[:,None]
  joint=torch.stack((after.sum(1)/nden,after.sum(2)/nden,
     torch.matmul(after.transpose(1,2),onbits.unsqueeze(-1)).squeeze(-1)/nden,
     torch.matmul(after,onbits.unsqueeze(-1)).squeeze(-1)/nden),dim=-1)
  n_logits=self.node_out(torch.cat((refined,joint,z[:,None,:].expand(-1,n,-1)),dim=-1)).squeeze(-1)
  return e_logits,n_logits

