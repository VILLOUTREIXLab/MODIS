# MODIS: Multi-omics Data Integration for Small and unpaired datasets

MODIS performs diagonal integration on unpaired samples by learning a probabilistic coupling of the heterogeneous data modalities into a shared latent space, leveraging class-labels to align modalities despite class imbalance and data scarcity.

##  Installation

It is recommended to use a virtual environment for this project. You can create and activate one with:

```bash
python3 -m venv env
source env/bin/activate
```

After activating the virtual environment, navigate into the `modis` foldier in the downloaded repository, and install MODIS with:

```bash
pip install .
```

## Citation

If you use MODIS in your research, please cite our paper:

Daniel Lepe-Soltero, Thierry Artières, Anaïs Baudot, Paul Villoutreix. MODIS: Multi-Omics Data Integration for Small and Unpaired Datasets (2025). https://doi.org/10.48550/arXiv.2503.18856.

```
@article{Lepe-Soltero2025,
  title={MODIS: Multi-Omics Data Integration for Small and Unpaired Datasets},
  author={Daniel Lepe-Soltero, Thierry Artières, Anaïs Baudot, Paul Villoutreix},
  journal={arXiv},
  pages={},
  year={2025},
  publisher={}
}
```

## Contributing

If you have a suggestion, find a bug, or would like to contribute, please open an issue or submit a pull request.