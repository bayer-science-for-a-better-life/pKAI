import argparse
from os.path import dirname, isfile, abspath
from protein import Protein
import torch


def download_model(file_name: str, file_path: str) -> None:
    from urllib import request, error
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    url = f"https://filedn.com/lcmXXv3bIOvYUloTyvVflnk/{file_name}"
    try:
        print("Downloading file...")
        request.urlretrieve(url, file_path)
    except error.HTTPError:
        raise error.HTTPError(
            f"No {file_name} found. Please check the model name you are using is supported (pKAI or pKAI+)."
        )
    except error.URLError:
        raise error.URLError(
            f"{url} not reachable. Please check that you have internet access."
        )


def load_model(model_name: str, device):
    path = dirname(abspath(__file__))
    fname = f"{model_name}_model.pt"
    fpath = f"{path}/{fname}"

    if not isfile(fpath):
        download_model(fname, fpath)

    device = torch.device(device)
    model = torch.jit.load(f"{fpath}").to(device)

    return model


def pKAI(pdb, model_name="pKAI", device="cpu", threads=None):
    if threads:
        torch.set_num_threads(threads)
    model = load_model(model_name, device)
    prot = Protein(pdb)
    prot.apply_cutoff()
    pks = prot.predict_pkas(model, device)
    return pks


def main():
    parser = argparse.ArgumentParser(description="pKAI")
    parser.add_argument("pdb_file", type=str, help="PDB file")
    parser.add_argument(
        "--model",
        type=str,
        choices=["pKAI", "pKAI+"],
        help="Number of threads to use",
        default="pKAI",
    )
    parser.add_argument(
        "--device", type=str, help="Device on which to run the model on", default="cpu"
    )
    parser.add_argument("--threads", type=str, help="Number of threads to use")

    args = parser.parse_args()

    pKAI(args.pdb_file, model_name=args.model, device=args.device, threads=args.threads)


if __name__ == "__main__":
    main()

