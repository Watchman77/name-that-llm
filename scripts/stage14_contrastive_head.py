"""Paired CPU heads with one loss change on identical fit-only prompt batches."""

from __future__ import annotations

import numpy as np


CLASSES = 18
PROJECTION = 64
PROMPTS_PER_BATCH = 8
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
TEMPERATURE = 0.1
CONTRASTIVE_WEIGHT = 0.2
THREADS = 2


def require(condition, code):
    if not condition:
        raise ValueError(code)


def pair_masks(labels, prompt_codes):
    """Positives cross prompts; negatives compete within the same prompt."""
    labels, prompt_codes = np.asarray(labels), np.asarray(prompt_codes)
    require(labels.ndim == 1 and labels.shape == prompt_codes.shape, "contrastive_pair_axes")
    same_label = labels[:, None] == labels[None, :]
    same_prompt = prompt_codes[:, None] == prompt_codes[None, :]
    return same_label & ~same_prompt, same_prompt & ~same_label


def prompt_schedule(prompt_codes, seed):
    """Return one frozen row schedule for both arms, containing full prompt groups."""
    groups = np.unique(prompt_codes)
    require(len(groups) >= 2, "contrastive_fit_prompt_count")
    rows = {p: np.flatnonzero(prompt_codes == p) for p in groups}
    batch_count = (len(groups) + PROMPTS_PER_BATCH - 1) // PROMPTS_PER_BATCH
    rng = np.random.default_rng(seed)
    return [tuple(np.concatenate([rows[p] for p in block])
                  for block in np.array_split(rng.permutation(groups), batch_count))
            for _ in range(EPOCHS)]


def prepare_fit_score(x_fit, x_score):
    """Fit standardization on the outer training fold only."""
    from sklearn.preprocessing import StandardScaler

    x_fit, x_score = np.asarray(x_fit), np.asarray(x_score)
    require(x_fit.ndim == x_score.ndim == 2 and x_fit.shape[1] == x_score.shape[1]
            and x_fit.shape[1] > 0 and len(x_fit) > 0 and len(x_score) > 0,
            "contrastive_feature_axes")
    require(all(x.dtype.kind in "fiu" and np.isfinite(x).all() for x in (x_fit, x_score)),
            "contrastive_nonfinite_input")
    scaler = StandardScaler().fit(x_fit)
    values = tuple(np.ascontiguousarray(scaler.transform(x), dtype=np.float32) for x in (x_fit, x_score))
    require(all(np.isfinite(x).all() for x in values), "contrastive_nonfinite_standardization")
    return values


def fit_pair(x_fit, y_fit, prompt_fit, x_score, seed):
    """Return finite CE and CE+contrastive logits in integer class order 0..17.

    No scored labels or prompt IDs are accepted. Only fitting prompts enter the
    optimizer, normalization fit, or contrastive pair construction.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F

    y, prompts = np.asarray(y_fit), np.asarray(prompt_fit)
    require(isinstance(seed, (int, np.integer)) and not isinstance(seed, bool)
            and 0 <= int(seed) < 2**32, "contrastive_seed")
    seed = int(seed)
    require(y.ndim == 1 and prompts.shape == y.shape and y.dtype.kind in "iu"
            and prompts.dtype.kind in "UO" and all(isinstance(p, str) and p for p in prompts)
            and np.array_equal(np.unique(y), np.arange(CLASSES)), "contrastive_training_axes")
    fit, score = prepare_fit_score(x_fit, x_score)
    require(len(fit) == len(y), "contrastive_fit_alignment")
    _, group_codes = np.unique(prompts, return_inverse=True)
    require(len(np.unique(group_codes * CLASSES + y)) == len(y), "contrastive_duplicate_prompt_class")
    batches = prompt_schedule(group_codes, seed)
    fit_tensor, score_tensor = torch.from_numpy(fit), torch.from_numpy(score)
    labels = torch.from_numpy(y.astype(np.int64, copy=False))

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(fit.shape[1], PROJECTION)
            self.output = nn.Linear(PROJECTION, CLASSES)

        def forward(self, values):
            embedded = F.normalize(self.projection(values), p=2.0, dim=1, eps=1e-12)
            return self.output(embedded), embedded

    def fit_arm(weight):
        # Initialize both arms bit-for-bit identically and do not consume the
        # caller's CPU RNG state. The optimizer and schedule are also shared.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model = Head()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                                      foreach=False, fused=False)
        model.train()
        for epoch in batches:
            for batch in epoch:
                indices = torch.from_numpy(batch)
                logits, embedded = model(fit_tensor.index_select(0, indices))
                loss = F.cross_entropy(logits, labels.index_select(0, indices))
                if weight:
                    p, n = pair_masks(y[batch], group_codes[batch])
                    require(p.any(axis=1).all() and n.any(axis=1).all(), "contrastive_batch_missing_pair")
                    positive, candidates = torch.from_numpy(p), torch.from_numpy(p | n)
                    sim = (embedded @ embedded.T) / TEMPERATURE
                    log_denom = sim.masked_fill(~candidates, -torch.inf).logsumexp(dim=1)
                    mean_positive = sim.masked_fill(~positive, 0.0).sum(dim=1) / positive.sum(dim=1)
                    loss = loss + weight * (log_denom - mean_positive).mean()
                require(bool(torch.isfinite(loss)), "contrastive_nonfinite_loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        model.eval()
        with torch.inference_mode():
            values = model(score_tensor)[0].numpy().astype(np.float64)
        require(values.shape == (len(score), CLASSES) and np.isfinite(values).all(),
                "contrastive_invalid_logits")
        return values

    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(THREADS)
        torch.use_deterministic_algorithms(True)
        return fit_arm(0.0), fit_arm(CONTRASTIVE_WEIGHT)
    finally:
        torch.use_deterministic_algorithms(old_deterministic, warn_only=old_warn_only)
        torch.set_num_threads(old_threads)
