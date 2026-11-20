import numpy as np
import pandas as pd

def build_event_hypergraph(df, entity_col, relation_col=None, max_entities=5000):
    """
    Builds a sparse-style incidence matrix H [events x entities].
    Entity co-occurrence groups are derived from observed records; they are NOT
    claimed to be ground-truth hyperedge annotations.
    """
    if entity_col not in df.columns:
        raise ValueError(f"Configured entity_col '{entity_col}' not found.")
    entities = df[entity_col].fillna("UNKNOWN").astype(str)
    unique = pd.Index(entities.unique()[:max_entities])
    mapping = {v:i for i,v in enumerate(unique)}
    H = np.zeros((len(df), len(unique)), dtype=np.float32)
    for r, entity in enumerate(entities):
        if entity in mapping: H[r, mapping[entity]] = 1.0
    relations = (df[relation_col].fillna("unknown").astype(str).to_numpy()
                 if relation_col and relation_col in df.columns else np.array(["all"]*len(df)))
    return H, unique.tolist(), relations

def aggregate_hypergraph(X, H):
    # Event-to-entity aggregation and entity-to-event propagation.
    degree = np.maximum(H.sum(axis=0, keepdims=True), 1.0)
    entity_repr = (H.T @ X) / degree.T
    event_degree = np.maximum(H.sum(axis=1, keepdims=True), 1.0)
    return (H @ entity_repr) / event_degree
