import torch
import os


def get_rmvpe(model_path=None, device=torch.device("cpu")):
    if model_path is None:
        assets_root = os.getenv('assets_root', 'assets')
        model_path = os.path.join(assets_root, 'rmvpe', 'rmvpe.pt')
    from infer.lib.rmvpe import E2E

    model = E2E(4, 1, (2, 2))
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    model = model.to(device)
    return model
