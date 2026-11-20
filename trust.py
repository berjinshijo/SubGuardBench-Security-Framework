import numpy as np
import pandas as pd

def evidence_weighted_trust(threat_probability, hyperedge_suspiciousness=None,
                            disagreement=None, agent_risk=None, log_risk=None):
    n=len(threat_probability)
    def arr(v): return np.zeros(n) if v is None else np.clip(np.asarray(v,dtype=float),0,1)
    risk = (0.40*arr(threat_probability) + 0.20*arr(hyperedge_suspiciousness)
            + 0.15*arr(disagreement) + 0.15*arr(agent_risk) + 0.10*arr(log_risk))
    return np.clip(1.0-risk,0,1)

def operational_safety_mapping(trust, criticality=None, essential=None,
                                trust_floor=0.35):
    n=len(trust)
    criticality=np.zeros(n) if criticality is None else np.asarray(criticality,dtype=float)
    essential=np.zeros(n,dtype=bool) if essential is None else np.asarray(essential,dtype=bool)
    actions=[]
    for i,t in enumerate(trust):
        if t >= 0.75: a="ALLOW"
        elif t >= 0.50: a="RESTRICT"
        elif essential[i] and t >= trust_floor: a="READ_ONLY"
        elif criticality[i] >= 0.8: a="REVIEW"
        else: a="DENY"
        actions.append(a)
    return actions
