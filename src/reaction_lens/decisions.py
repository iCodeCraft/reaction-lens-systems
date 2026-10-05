"""Complete all-option decision output; scores are not probabilities."""
import numpy as np
import torch

def validate_options(options):
    if not options:
        raise ValueError("Supply at least one reaction option")
    for row in options:
        if not isinstance(row, dict) or not isinstance(row.get("reaction_id"),str) or not row["reaction_id"].strip() or not isinstance(row.get("description"),str) or not row["description"].strip():
            raise ValueError("Every option needs nonempty reaction_id and description strings")
    if len({r["reaction_id"] for r in options}) != len(options):
        raise ValueError("Duplicate option IDs")

def decision_payload(options, logits, threshold):
    validate_options(options)
    logits = np.asarray(logits, dtype=np.float32)
    if logits.shape != (len(options),) or not np.isfinite(logits).all() or not np.isfinite(threshold):
        raise ValueError("Invalid logits/threshold")
    scores = torch.sigmoid(torch.from_numpy(logits)).numpy()
    order = np.argsort(-logits,kind="stable")
    ranks = np.empty(len(options),dtype=int); ranks[order] = np.arange(1,len(options)+1)
    decisions = [{"reaction_id": row["reaction_id"], "description": row["description"],
                  "logit": float(logits[i]), "sigmoid_score": float(scores[i]),
                  "selected": bool(logits[i] >= threshold), "rank": int(ranks[i])} for i,row in enumerate(options)]
    return {"options_scored": len(options), "raw_logit_threshold": float(threshold),
            "proposed_reaction_ids": [r["reaction_id"] for r in decisions if r["selected"]],
            "options": decisions, "scope": "Every supplied option scored independently; no top-k selection limit. Scores are not calibrated probabilities."}
