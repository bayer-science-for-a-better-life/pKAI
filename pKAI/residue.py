from atom import Atom
import torch

AA_ATOMS = {
    "ASP": ["OD1", "OD2"],
    "CTR": ["OD1", "OXT"],
    "CYS": ["SG"],
    "GLU": ["OE1", "OE2"],
    "HIS": ["ND1", "NE2"],
    "LYS": ["NZ"],
    "NTR": ["N"],
    "TYR": ["OH"],
}

ATOM_OHE = [
    "OD1_ASN",
    "NE2_HIS",
    "N",
    "NE1_TRP",
    "NH_ARG",
    "NZ_LYS",
    "O_COOH",
    "NE2_GLN",
    "O",
    "OXT",
    "OH_TYR",
    "SD_MET",
    "OE1_GLN",
    "ND1_HIS",
    "OG1_THR",
    "ND2_ASN",
    "OG_SER",
    "NE_ARG",
    "SG_CYS",
]

RES_OHE = ["CTR", "HIS", "ASP", "LYS", "GLU", "CYS", "NTR", "TYR"]


class Residue:
    def __init__(self, protein, chain, resname, resnumb):

        self.protein = protein
        self.chain = chain
        self.resname = resname
        self.resnumb = resnumb
        self.atoms = {}

        self.env_dists = []
        self.env_anames = []
        self.env_resnames = []

        self.env_oheclasses = []
        self.dists_sorted = []
        self.atoms_sorted = []

        self.env_tensor_size = 250
        self.ohe_size = len(ATOM_OHE)
        self.env_tensor = torch.zeros(
            self.env_tensor_size, 1, self.ohe_size, dtype=torch.float32
        )
        self.ohe_resname = torch.zeros(1, 1, len(RES_OHE), dtype=torch.float32)
        self.input_layer = []

    def add_atom(self, aname, anumb, x, y, z):
        atom = Atom(aname, anumb, x, y, z, self)
        self.atoms[anumb] = atom

    def iter_atoms(self):
        for atom in self.atoms.values():
            yield atom

    def calc_cutoff_atoms(self, cutoff_dist: float):

        cutoff_sq = cutoff_dist ** 2

        site_ref_atoms = []
        for atom in self.iter_atoms():
            if atom.aname in AA_ATOMS[self.resname]:
                site_ref_atoms.append(atom)

        for atom in self.protein.iter_atoms():
            min_dist = 99999
            for residue_atom in site_ref_atoms:

                if atom in self.atoms.values():
                    continue

                sqdist = residue_atom.calc_sqdist(atom)

                if sqdist < min_dist:
                    min_dist = sqdist

            if min_dist < cutoff_sq:
                self.env_anames.append(atom.aname)
                self.env_resnames.append(atom.residue.resname)
                self.env_dists.append(min_dist ** 0.5)

    def encode_atoms(self):
        for aname, resname in zip(self.env_anames, self.env_resnames):
            aname = aname.strip()
            resname = resname.strip()
            if aname == "SE" and resname == "SEC":
                continue
            if aname in ("N", "O", "OXT"):
                new_aname = f"{aname}"
            elif aname in ("OD1", "OD2", "OE1", "OE2") and resname in (
                "ASP",
                "GLU",
                "UNK",
            ):
                new_aname = f"O_COOH"
            elif aname in ("NH1", "NH2") and resname == "ARG":
                new_aname = f"NH_ARG"
            else:
                new_aname = f"{aname}_{resname}"

            self.env_oheclasses.append(new_aname)

    def input_sort(self):
        for dist, atom in sorted(zip(self.env_dists, self.env_oheclasses)):
            self.dists_sorted.append(dist)
            self.atoms_sorted.append(atom)

    def apply_ohe(self):
        for li, letter in enumerate(self.atoms_sorted):
            if li == self.env_tensor_size:
                break

            ohe_index = ATOM_OHE.index(letter)
            r = self.dists_sorted[li]

            self.env_tensor[li][0][ohe_index] = 1 / (r ** 2)

        ohe_index = RES_OHE.index(self.resname)
        self.ohe_resname[0][0][ohe_index] = 1

        self.input_layer = torch.cat(
            (self.env_tensor.view(-1), self.ohe_resname.view(-1)), dim=0
        )

    def encode_input(self):

        self.encode_atoms()
        self.input_sort()
        self.apply_ohe()

