from typing import Any, Dict, Optional


def get_grouped_model_parameters(
    model, base_params: Optional[Dict[str, Any]] = None, **name2params
):
    if base_params is None:
        base_params = {}

    grouped_parameters = []

    seen_parameter_names = set()
    for name, params in name2params.items():
        group_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in n:
                if n in seen_parameter_names:
                    raise ValueError(f"{n} appears more than one time")
                group_params.append(p)
                seen_parameter_names.add(n)
        grouped_parameters.append({"params": group_params, **params})

    if base_params:
        grouped_parameters.append(
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (n not in seen_parameter_names) and p.requires_grad
                ],
                **base_params,
            }
        )

    return grouped_parameters
