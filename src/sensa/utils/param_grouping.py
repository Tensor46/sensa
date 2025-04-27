from typing import Any

import torch


def base_param_grouping(model: torch.nn.Module, *, groups: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Helper to split `model` parameters for optimizer weight-decay settings.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose parameters to inspect.
    groups : list of dict, optional
        An existing list to append to; if `None`, two new empty groups are created.

    Returns
    -------
    list of dict
        `[{"params": [...]}, {"params": [...]}]`. The first group holds
        weight-decay parameters; the second holds parameters exempt from decay.
    """
    groups = [{"params": []}, {"params": []}] if groups is None else groups
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_wdecay = "token" in name or p.ndim <= 1
        groups[int(no_wdecay)]["params"].append(p)
    return groups


def merge_param_groups(groups: list[dict[str, Any]], groups_new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge two lists of parameter-group dictionaries, aligning and concatenating `"params"` lists.

    Ensures both `groups` and `groups_new` have the same length by
    prepending empty groups as needed, then extends each group's
    `"params"` list with those from the corresponding new group.

    Parameters
    ----------
    groups : list of dict[str, Any]
        Existing parameter groups (each dict must have a `"params"` key).
    groups_new : list of dict[str, Any]
        New parameter groups to merge into `groups`.

    Returns
    -------
    list of dict[str, Any]
        A unified list of groups where each group's `"params"` contains
        the concatenation of the original and new parameters.
    """
    if isinstance(groups, list) and len(groups) == 0:
        groups = [{"params": []}, {"params": []}]

    if len(groups) > len(groups_new):
        groups_new = [{"params": []} for _ in range(len(groups) - len(groups_new))] + groups_new
    if len(groups) < len(groups_new):
        groups = [{"params": []} for _ in range(len(groups_new) - len(groups))] + groups
    for group, group_new in zip(groups, groups_new, strict=False):
        group["params"] += group_new["params"]
    return groups
