from collections import OrderedDict


def filter_state_dict(state_dict: OrderedDict, tag: str) -> OrderedDict:
    """Extract and rename a subset of parameters from a checkpoint or raw state_dict.

    Parameters
    ----------
    state_dict : OrderedDict
        Either a raw model state_dict (mapping parameter names → tensors), or a
        checkpoint dict containing a `"state_dict"` entry.
    tag : str
        The prefix used to filter keys (e.g. `"encoder."`). Only keys starting with
        this prefix are retained, and then the prefix is removed from their names.

    Returns
    -------
    OrderedDict
        A new state_dict containing only the parameters whose names began with
        `tag`, with `tag` removed from each key.

    Examples
    --------
    >>> # Suppose `ckpt` is loaded via torch.load and contains both encoder and decoder:
    >>> ckpt = torch.load("multimodule.ckpt")
    >>> state_dict = filter_state_dict(ckpt, tag="encoder.")
    >>> model.encoder.load_state_dict(state_dict)
    """
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    invalid = [key for key in state_dict if not key.startswith(tag)]
    for key in invalid:
        del state_dict[key]

    valid = list(state_dict.keys())
    for key in valid:
        state_dict[key.removeprefix(tag)] = state_dict[key]
        del state_dict[key]

    return state_dict
