import random
from collections import deque
import numpy as np
import torch
from torch import nn

ACTIONS = ["allow", "restrict", "isolate", "deny", "revert"]

class SubstationSafetyEnv:
    """Controlled toy simulator for policy learning; not measured dataset transitions."""
    def __init__(self, n_entities=20, max_steps=200, seed=42):
        self.rng=np.random.default_rng(seed); self.n=n_entities; self.max_steps=max_steps
        self.action_space=len(ACTIONS); self.reset()
    def reset(self):
        self.t=0
        self.threat=self.rng.random(self.n)
        self.critical=self.rng.random(self.n)
        self.essential=self.rng.random(self.n)<0.2
        self.restricted=np.zeros(self.n,dtype=float)
        return self._state()
    def _state(self):
        return np.stack([self.threat,self.critical,self.essential.astype(float),self.restricted],axis=1).astype(np.float32)
    def valid_mask(self):
        # Never isolate an essential entity; deny is also masked for essential entities.
        return np.stack([np.ones(self.n), np.ones(self.n),
                         (~self.essential).astype(float),
                         (~self.essential).astype(float), np.ones(self.n)],axis=1)
    def step(self, entity, action):
        self.t+=1
        if self.valid_mask()[entity,action] == 0:
            reward=-5.0
        else:
            risk=self.threat[entity]
            if action==0: reward=1.0-5*risk
            elif action==1: reward=2.0*risk-0.3
            elif action==2: reward=3.0*risk-1.0-2*self.critical[entity]
            elif action==3: reward=2.5*risk-0.7
            else: reward=0.1
            self.restricted[entity]=float(action in (1,2,3))
        self.threat=np.clip(self.threat+self.rng.normal(0,0.015,self.n),0,1)
        done=self.t>=self.max_steps
        return self._state(),float(reward),done,{"masked":bool(self.valid_mask()[entity,action]==0)}

class QNet(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__(); self.net=nn.Sequential(nn.Linear(state_dim,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,n_actions))
    def forward(self,x): return self.net(x)

def train_saam_dqn(episodes=100,max_steps=200,seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    env=SubstationSafetyEnv(max_steps=max_steps,seed=seed)
    online=QNet(4,len(ACTIONS)); target=QNet(4,len(ACTIONS)); target.load_state_dict(online.state_dict())
    opt=torch.optim.Adam(online.parameters(),lr=1e-3); memory=deque(maxlen=10000)
    rewards=[]; gamma=.95; eps=1.0
    for ep in range(episodes):
        state=env.reset(); total=0
        for _ in range(max_steps):
            entity=random.randrange(env.n); s=torch.tensor(state[entity])
            mask=env.valid_mask()[entity].astype(bool)
            if random.random()<eps: action=random.choice(np.flatnonzero(mask).tolist())
            else:
                with torch.no_grad():
                    q=online(s); q[~torch.tensor(mask)]=-1e9; action=int(q.argmax())
            nxt,r,done,_=env.step(entity,action); memory.append((state[entity],action,r,nxt[entity],done,env.valid_mask()[entity]))
            state=nxt; total+=r
            if len(memory)>=64:
                batch=random.sample(memory,64)
                ss,aa,rr,ns,dd,mm=zip(*batch)
                ss=torch.tensor(np.array(ss)); ns=torch.tensor(np.array(ns)); aa=torch.tensor(aa)
                rr=torch.tensor(rr,dtype=torch.float32); dd=torch.tensor(dd,dtype=torch.float32); mm=torch.tensor(np.array(mm),dtype=torch.bool)
                q=online(ss).gather(1,aa[:,None]).squeeze()
                with torch.no_grad():
                    nq=online(ns).masked_fill(~mm,-1e9).argmax(1)
                    target_q=target(ns).gather(1,nq[:,None]).squeeze()
                    y=rr+gamma*(1-dd)*target_q
                loss=((q-y)**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
            if done: break
        eps=max(.05,eps*.985); rewards.append(total)
        if (ep+1)%10==0: target.load_state_dict(online.state_dict())
    return online,rewards
