import yaml
import numpy as np
from pathlib import Path
from src.subguard.data import load_and_validate
from src.subguard.preprocessing import chronological_split, MultimodalPreprocessor
from src.subguard.hypergraph import build_event_hypergraph, aggregate_hypergraph
from src.subguard.model import train_model
from src.subguard.trust import evidence_weighted_trust, operational_safety_mapping
from src.subguard.saam_dqn import train_saam_dqn
from src.subguard.evaluation import evaluate_classifier

def main():
    cfg=yaml.safe_load(open("config.yaml",encoding="utf-8"))
    df=load_and_validate(cfg)
    d=cfg["data"]; p=cfg["preprocessing"]; m=cfg["model"]
    label=d["label_col"]
    if label not in df.columns:
        raise ValueError(f"Label column '{label}' not found. Set data.label_col in config.yaml to the real target column.")
    train,val,test=chronological_split(df,d.get("timestamp_col"),p["test_fraction"],p["validation_fraction"])
    prep=MultimodalPreprocessor(d.get("numeric_cols",[]),d.get("categorical_cols",[]),
        d.get("text_col"),p["text_features"],p["top_k_features"])
    Xtr=prep.fit_transform(train,train[label].to_numpy())
    Xva=prep.transform(val); Xte=prep.transform(test)
    # Build each split independently to avoid cross-split entity leakage.
    Htr,_,_=build_event_hypergraph(train,d["entity_col"],d.get("relation_col"))
    Hva,_,_=build_event_hypergraph(val,d["entity_col"],d.get("relation_col"))
    Hte,_,_=build_event_hypergraph(test,d["entity_col"],d.get("relation_col"))
    Gtr=aggregate_hypergraph(Xtr,Htr); Gva=aggregate_hypergraph(Xva,Hva); Gte=aggregate_hypergraph(Xte,Hte)
    model,_,classes=train_model(Xtr,Gtr,train[label].to_numpy(),Xva,Gva,val[label].to_numpy(),
        epochs=m["epochs"],batch_size=m["batch_size"],lr=m["learning_rate"])
    import torch
    device=next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        logits,_=model(torch.tensor(Xte,device=device),torch.tensor(Gte,device=device))
        probs=torch.softmax(logits,dim=1).cpu().numpy()
    pred=classes[probs.argmax(1)]
    metrics=evaluate_classifier(test[label].to_numpy(),pred,probs,classes,cfg["evaluation"]["output_dir"])
    threat=probs[:,min(1,probs.shape[1]-1)]
    trust=evidence_weighted_trust(threat)
    actions=operational_safety_mapping(trust)
    Path("results").mkdir(exist_ok=True)
    import pandas as pd
    pd.DataFrame({"threat_probability":threat,"trust_score":trust,"safety_action":actions}).to_csv("results/event_risk_and_actions.csv",index=False)
    # RL is trained in an explicitly separate simulator, not on fabricated dataset transitions.
    _,rewards=train_saam_dqn(cfg["rl"]["episodes"],cfg["rl"]["max_steps"],cfg["rl"]["seed"])
    pd.DataFrame({"episode":range(1,len(rewards)+1),"cumulative_reward":rewards}).to_csv("results/saam_dqn_rewards.csv",index=False)
    print("Completed baseline pipeline. Metrics:",metrics)
    print("NOTE: This is an adaptable baseline scaffold. Validate feature mappings, hyperedge semantics, temporal windows, and RL simulator against the dataset/paper design.")

if __name__=="__main__": main()
