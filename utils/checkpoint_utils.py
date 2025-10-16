import os
import torch


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer state tensors to the specified device."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def load_model(
    simulator: torch.nn.Module,
    model_dir: str,
    model_file: str,
    train_state_file: str,
    device: torch.device,
):
    """Load model weights and optimizer state; return (simulator, step, optimizer).

    Expects model_dir to end with a path separator or be joined externally.
    """
    model_path = model_dir + model_file
    state_path = model_dir + train_state_file

    if os.path.exists(model_path) and os.path.exists(state_path):
        # load model
        simulator.load(model_path)

        # load train state
        train_state = torch.load(state_path)
        # set optimizer state
        optimizer = torch.optim.Adam(simulator.parameters())
        optimizer.load_state_dict(train_state["optimizer_state"])
        optimizer_to(optimizer, device)
        # set global train state
        step = train_state["global_train_state"].pop("step")
        return simulator, step, optimizer

    msg = f"Specified model_file {model_path} and train_state_file {state_path} not found."
    raise FileNotFoundError(msg)


