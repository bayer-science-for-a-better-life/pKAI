# pKAI

A fast and interpretable deep learning approach to accurate electrostatics-driven pKa prediction

```
@article{pkai,
author = {Reis, Pedro B. P. S. and Vila-Viçosa, Diogo and Bertolini, Marco and Montanari, Floriane and Machuqueiro, Miguel and Clevert, Djork-Arné},
title = {pKAI: A fast and interpretable deep learning approach to accurate electrostatics-driven pKa prediction},
note = {in preparation}
}
```

### Installation & Basic Usage

We recommend installing pKAI on a conda enviroment. The pKAI+ model will be downloaded on the first execution and saved for subsequent runs.

```
python3 -m pip install pKAI

pKAI <pdbfile>
```

It can also be used as python function,
```
from pKAI.pKAI import pKAI

pks = pKAI(pdb)
```
where each element of the returned list is a tuple of size 4. (chain, resnumb, resname, pk)

## pKAI+ vs pKAI models

pKAI+ (default model) aims to predict experimental p<i>K</i><sub>a</sub> values from a single conformation. To do such, the interactions characterized in the input structure are given less weight and, as a consequence, the predictions are closer to the p<i>K</i><sub>a</sub> values of the residues in water. This effect is comparable to an increase in the dielectric constant of the protein in Poisson-Boltzmann models. In these models, the dielectric constant tries to capture, among others, electronic polarization and side-chain reorganization. When including conformational sampling explicitly, one should use a lower value for the dielectric constant of the protein. Likewise, one should use pKAI -- instead of pKAI+ -- as in this model there is no penalization of the interactions' impact on the predicted p<i>K</i><sub>a</sub> values.

tl;dr version
- use pKAI+ for p<i>K</i><sub>a</sub> predictions arising from a single structure
- use pKAI for p<i>K</i><sub>a</sub> predictions arising from multiple conformations

Change the model to be used in the calculation by evoking the `model` argument:
```
pKAI <pdbfile> --model pKAI
```

## Benchmark

Performed on 927 experimental values taken from the PKAD database.

| Method                | RMSE | MAE  |
|-----------------------|------|------|
| Thurlkill<sup>1</sup> | 1.02 | 0.70 |
| PROPKA<sup>2</sup>    | 1.04 | 0.69 |
| PypKa<sup>3</sup>     | 0.98 | 0.66 |
| pKAI+                 | 0.91 | 0.59 |

[1] Thurlkill, Richard L et al. “pK values of the ionizable groups of proteins.” doi:<a href="https://doi.org/10.1110/ps.051840806">10.1110/ps.051840806</a>

[2] Olsson, Mats H M et al. “PROPKA3: Consistent Treatment of Internal and Surface Residues in Empirical pKa Predictions.” doi:<a href="https://doi.org/10.1021/ct100578z">10.1021/ct100578z</a>

[3] Reis, Pedro B P S et al. “PypKa: A Flexible Python Module for Poisson-Boltzmann-Based pKa Calculations.” doi:<a href="https://doi.org/10.1021/acs.jcim.0c00718">10.1021/acs.jcim.0c00718</a>


## License

This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

## Contacts
Please submit a github issue to report bugs and to request new features. Alternatively, you may <a href="pdreis@fc.ul.pt"> email the developer directly</a>.
