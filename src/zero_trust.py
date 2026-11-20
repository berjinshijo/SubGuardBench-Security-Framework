from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json

LEVELS=("FULL","RESTRICTED","READ_ONLY","DENIED")
@dataclass
class AccessRequest:
    identity: str
    trust: float
    suspiciousness: float
    operation: str
    criticality: float
    agent_risk: float=0.0
    essential: bool=False

def policy_decision(req, version="v1", previous="FULL"):
    if req.essential and req.trust>=0.35 and req.suspiciousness<=0.75:
        level="READ_ONLY" if req.trust<0.6 else "FULL"
    elif req.trust>=0.8 and req.suspiciousness<0.2 and req.agent_risk<0.2:
        level="FULL"
    elif req.trust>=0.55 and req.suspiciousness<0.5:
        level="RESTRICTED"
    elif req.trust>=0.35:
        level="READ_ONLY"
    else: level="DENIED"
    # Simple hysteresis: avoid a one-step upgrade from DENIED to FULL.
    order={v:i for i,v in enumerate(LEVELS)}
    if previous in order and level=="FULL" and previous=="DENIED": level="READ_ONLY"
    return level

def enforce_request(req, dependency_is_essential=False, audit_path="results/zero_trust_audit.jsonl"):
    decision=policy_decision(req)
    action={"FULL":"ALLOW","RESTRICTED":"RESTRICT","READ_ONLY":"READ_ONLY","DENIED":"DENY"}[decision]
    if dependency_is_essential and action in ("ISOLATE","DENY"): action="RESTRICT"
    record={"timestamp":datetime.now(timezone.utc).isoformat(),"request":asdict(req),
            "pdp_decision":decision,"pep_action":action,"policy_version":"v1"}
    with open(audit_path,"a",encoding="utf-8") as f: f.write(json.dumps(record)+"\n")
    return decision,action
