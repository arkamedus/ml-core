import torch

def encode_fixed_length_right_pad(
    vocab,
    text: str,
    *,
    ctx: int,
    pad_id: int,
    bos_id=None,
    eos_id=None,
    keep: str = "last",
):
    ids = vocab.tokenize(str(text))

    if bos_id is not None:
        ids = [bos_id] + ids
    if eos_id is not None:
        ids = ids + [eos_id]

    if len(ids) > ctx:
        if keep == "first":
            ids = ids[:ctx]
        elif keep == "last":
            ids = ids[-ctx:]
        else:
            raise ValueError("keep must be 'first' or 'last'")

    if len(ids) < ctx:
        ids = ids + [pad_id] * (ctx - len(ids))

    return torch.tensor(ids, dtype=torch.long)


def zero_pad_positions(x, pad_mask):
    if pad_mask is None:
        return x
    return x.masked_fill(pad_mask.unsqueeze(-1), 0.0)


def masked_mean_pool(x, pad_mask):
    nonpad = ~pad_mask
    x = zero_pad_positions(x, pad_mask)
    denom = nonpad.sum(dim=1, keepdim=True).clamp(min=1)
    return x.sum(dim=1) / denom


def last_nonpad_pool(x, pad_mask):
    nonpad = ~pad_mask
    lengths = nonpad.sum(dim=1).clamp(min=1)
    last_idx = lengths - 1
    return x[torch.arange(x.size(0), device=x.device), last_idx]