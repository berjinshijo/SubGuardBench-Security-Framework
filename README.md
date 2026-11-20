# SubGuardBench Security Framework — implementation scaffold

This repository converts the proposed eight-stage workflow into a modular Python project.

## Important reproducibility note
The IEEE DataPort page did not expose the downloadable dataset schema in this environment. Therefore this is a **schema-adaptable implementation scaffold**, not a claim that the code has already been validated on SubGuardBench. Do not report its output as research results until it has been run on the actual dataset and checked against its documentation.

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Place the downloaded dataset at `data/raw/subguardbench.csv` (or edit `data.input_path` in `config.yaml`).

## Configure first
Edit `config.yaml` using the real dataset schema:
- `timestamp_col`: timestamp column
- `label_col`: event-level ground-truth class/label
- `entity_col`: stable device/asset identifier
- `relation_col`: interaction/relation type if present
- `text_col`: agent text/tool-call description if present
- `numeric_cols` and `categorical_cols`: only columns that really exist

If SubGuardBench distributes four modalities as separate files, align them by timestamp and documented event/device keys before running the pipeline. Do not join only by row position.

## Run
```bash
python main.py
```

## Implemented stages
1. Dataset loading, schema display, timestamp parsing and chronological ordering.
2. Train-only RobustScaler, imputation, categorical encoding, HashingVectorizer, train-only mutual-information selection.
3. Event/entity incidence hypergraph and event-to-entity-to-event aggregation. Inferred relations must be described as derived.
4. Compact hypergraph-evidence classifier with low-rank feature mixing and an auxiliary event-risk head.
5. Evidence-weighted trust and operational safety mapping.
6. Double-DQN-style agent with operating-state action masking in a separate controlled simulator.
7. Attribute-based PDP/PEP decision logic with essential-operation protection and JSONL audit.
8. Detection metrics and confusion-matrix graph; RL reward log.

## Research-grade extensions still requiring dataset-specific design
- True four-modality timestamp alignment and modality-specific missingness handling.
- Explicit relation-specific sparse hyperedges (the generic builder is a starting point).
- Real temporal windows / sequence batching and cross-modal evidence-conditioned temporal mixing.
- Weakly supervised top-k multiple-instance hyperedge objective where hyperedge targets are justified.
- Calibrated threat probabilities and independently measured process/network disagreement.
- A validated power-substation transition simulator, reward specification and safety constraints for RL.
- A policy test harness that measures containment, false isolation, critical-link availability and violations.
- Baselines, ablations, repeated seeds, confidence intervals, parameter/FLOP/latency/memory reporting.

## Output files
- `results/metrics.json`
- `results/event_risk_and_actions.csv`
- `results/saam_dqn_rewards.csv`
- `results/graphs/confusion_matrix.png`

Never invent unavailable labels, modality fields, graph edges, or performance numbers.
