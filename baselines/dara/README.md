![DARA logo](https://github.com/CederGroupHub/dara/blob/main/logo/dara.jpg?raw=true)

# DARA: Data-driven automated Rietveld analysis for phase search and refinement
[![GitHub licence](https://img.shields.io/github/license/CederGroupHub/dara)](https://github.com/CederGroupHub/dara/blob/main/LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.1021%2Facs.chemmater.5c02820-blue)](https://doi.org/10.1021/acs.chemmater.5c02820)
[![Pytest](https://github.com/CederGroupHub/dara/actions/workflows/pytest.yaml/badge.svg?branch=main)](https://github.com/CederGroupHub/dara/actions/workflows/pytest.yaml)
[![Ruff Check](https://github.com/CederGroupHub/dara/actions/workflows/ruff.yaml/badge.svg?branch=main)](https://github.com/CederGroupHub/dara/actions/workflows/ruff.yaml)

Automated phase search with BGMN.
## Installation
```bash
pip install dara-xrd
```

For more details about installation, please refer to [installation guide](https://CederGroupHub.github.io/dara/install.html).

## Web Server
Dara ships with a browser-based web server for an out-of-box experience of Dara. To launch the webserver, run
```bash
dara server
```

Then you can open http://localhost:8898 to see an application that can submit, manage, and view jobs.


## Documentation
For more details about usage, please refer to the [documentation](https://CederGroupHub.github.io/dara/).

## Citation
If you use DARA in your research, please consider citing the following paper:

```
@article{doi:10.1021/acs.chemmater.5c02820,
  author = {Fei, Yuxing and McDermott, Matthew J. and Rom, Christopher L. and Wang, Shilong and Ceder, Gerbrand},
  title = {Dara: Automated Multiple-Hypothesis Phase Identification and Refinement from Powder X-ray Diffraction},
  journal = {Chemistry of Materials},
  volume = {38},
  number = {3},
  pages = {1364-1376},
  year = {2026},
  doi = {10.1021/acs.chemmater.5c02820},
  url = {https://doi.org/10.1021/acs.chemmater.5c02820},
  eprint = {https://doi.org/10.1021/acs.chemmater.5c02820}
}
```
