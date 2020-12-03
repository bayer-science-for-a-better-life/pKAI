import argparse
from os.path import dirname
from protein import Protein

from run import LitDist
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pKAI")
    parser.add_argument("pdb_file", type=str, help="PDB file")

    args = parser.parse_args()

    # torch.set_num_threads(4)

    path = dirname(__file__)

    device = torch.device("cpu")  # "cuda:3")

    # Pytorch Lightning Checkpoint
    # model = (
    #    LitDist().load_from_checkpoint(checkpoint_path=f"{path}/model.ckpt").to(device)
    # )
    # model.eval()

    # Pytorch State Dict
    # torch.save(model.state_dict(), f"{path}/model_state.pt")
    # model = LitDist().to(device)
    # model.load_state_dict(torch.load(f"{path}/model_state.pt"))
    # model.eval()

    # TorchScript
    # model = LitDist().to(device).half()
    # model.load_state_dict(torch.load(f"{path}/model_state_half.pt"))
    # model = model.half()
    # script = model.to_torchscript()
    # torch.jit.save(script, f"{path}/model_script_half.pt")

    model = torch.jit.load(f"{path}/model_script.pt")  # .to(device)

    prot = Protein(args.pdb_file)

    prot.apply_cutoff()

    prot.predict_pkas(model)

