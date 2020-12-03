from residue import Residue

PK_MODS = {
    "ASP": 3.79,
    "CTR": 2.90,
    "CYS": 8.67,
    "GLU": 4.20,
    "HIS": 6.74,
    "LYS": 10.46,
    "NTR": 7.99,
    "TYR": 9.59,
}


class Protein:
    def __init__(self, pdb_f):

        self.pdb_f = pdb_f
        self.tit_residues = {}
        self.all_residues = {}
        self.termini = {}  # TODO implement this later. ignored for now

        self.read_pdb()

    def iter_atoms(self):
        for residue in self.iter_residues():
            for atom in residue.iter_atoms():
                yield atom

    def iter_residues(self, titrable_only=False):
        if titrable_only:
            residues = self.tit_residues
        else:
            residues = self.all_residues
        for chain in sorted(residues.keys()):
            chain_residues = residues[chain]
            for resnumb in sorted(chain_residues.keys()):
                residue = chain_residues[resnumb]
                yield residue

    def read_pdb(self):
        """Removes all atoms from pdb_f that are not Nitrogens, Sulfurs, or Oxygens"""
        tit_aas = PK_MODS.keys()
        with open(self.pdb_f) as f:
            for line in f:
                if line.startswith("ATOM "):
                    line_cols = self.read_pdb_line(line)
                    (
                        aname,
                        anumb,
                        b,
                        resname,
                        chain,
                        resnumb,
                        x,
                        y,
                        z,
                        icode,
                    ) = line_cols

                    if b not in (" ", "A") or icode != " ":
                        continue

                    if aname[0] not in "NOS":
                        continue

                    if chain not in self.all_residues.keys():
                        self.all_residues[chain] = {}

                    if resnumb not in self.all_residues[chain].keys():
                        new_res = Residue(self, chain, resname, resnumb)
                        self.all_residues[chain][resnumb] = new_res
                    else:
                        new_res = self.all_residues[chain][resnumb]

                    atom = new_res.add_atom(aname, anumb, x, y, z)

                    if resname in tit_aas:
                        if chain not in self.tit_residues.keys():
                            self.tit_residues[chain] = {}

                        if resnumb not in self.tit_residues[chain]:
                            self.tit_residues[chain][resnumb] = new_res

    @staticmethod
    def read_pdb_line(line: str) -> tuple:
        aname = line[12:16].strip()
        anumb = int(line[5:11].strip())
        b = line[16]
        resname = line[17:21].strip()
        chain = line[21]
        resnumb = int(line[22:26])
        icode = line[26]
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        return (aname, anumb, b, resname, chain, resnumb, x, y, z, icode)

    def apply_cutoff(self, cutoff_dist=15):

        # TODO speed up by doing distance matrix a priori in cython

        for residue in self.iter_residues(titrable_only=True):
            residue.calc_cutoff_atoms(cutoff_dist)
            residue.encode_input()

    def predict_pkas(self, model, device):
        pks = []
        for residue in self.iter_residues(titrable_only=True):
            x = residue.input_layer
            resname = residue.resname

            dpk = float(model(x.to(device)))
            pk = round(dpk + PK_MODS[resname], 2)

            to_print = (
                f"{residue.chain:4} {residue.resnumb:6} {residue.resname:4} {pk:5.2f}"
            )
            print(to_print)

            result = (residue.chain, residue.resnumb, residue.resname, pk)
            pks.append(result)
        return pks

