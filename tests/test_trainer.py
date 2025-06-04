import torch

import sensa

from . import utils


def test_mae():
    sensa.trainer.ssl.MAE.__dataset__.__dryrun__ = True
    pt_model = sensa.trainer.ssl.MAE(**sensa.assets.load_yaml(utils.PATH_TO_TESTS / "samples/mae_vit.yaml"))

    # unpack batch
    images = torch.stack([pt_model.data[i][0] for i in range(4)])
    out = pt_model.encoder.forward_features(images)
    o_encoded = out["features"]
    # decode to reconstruct all patches
    _ = pt_model.decoder(o_encoded)
