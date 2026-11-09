# ReMiST

ReMiST reuses fixed historical and spatial features during monthly ENSO forecasting. It combines a multi-branch historical and spatial encoder (MHSE), variable-wise memory (VMIR), context-conditioned evidence readout (RCER), and seasonal tendency–innovation recursion (STIR).

## Artifact contents

- `source/remist_confirmatory_20261107` contains the ReMiST training recipe, shared-encoder heads, input packet assembly, and the whole-module STIR, VMIR, and RCER ablations. Sibling directories contain the inherited model components. Directory names retain the run identifiers for provenance.
- `tools/score_checkpoint.py` evaluates an individual checkpoint on the paper's SODA and GODAS cohort.
- `protocol/` contains selected learning-rate multipliers and input packet checksums. `results/` contains the four tables from the final paper for reference.
- The repository supplies the ReMiST code and paper tables. Exact retraining and checkpoint scoring require the prepared inputs and initializers listed below. Third-party baseline retraining, held-out fold preparation, timing and plotting pipelines are outside this snapshot.

## Environment and initialization

The reported experiments used Python 3.12.12, PyTorch 2.9.1+cu130, NumPy 2.3.5, CUDA 13.0 and NVIDIA RTX A6000 GPUs. Install the matching CUDA-enabled PyTorch wheel, then install `requirements.txt`.

From the repository root, fetch the pinned upstream DPOT source and checksum-verified DPOT-Tiny checkpoint:

```bash
python source/enso_route_b_mechanism_20261029/operator/fetch.py
```

The study also uses a task-specific initialization file, `source/enso_ema_ablation_20261031/rebuild_inputs/dpot_pretrained_interface.pt`. The public DPOT-Tiny checkpoint alone does not reconstruct this interface. The retained Mamba3 source is covered by the adjacent Apache-2.0 license. DPOT is obtained from its upstream repository.

## Data and split

The regional inputs are 24 months of eight-dimensional ocean histories. The gridded input is 12 months of 24 × 80 sea-surface temperature fields. Each forecast produces 12 monthly states. The Niño 3.4 index is state coordinate 8. Scores use output positions 2–12, corresponding to index center leads 1–11.

| Use | Product | Nominal issue months |
| --- | --- | --- |
| Train | CESM2, FIO-ESM-2-0, E3SM-1-1 | Jan 1870–Dec 1980 for each model |
| Validation | GFDL-CM4 | Jan 1985–Dec 2013 |
| Evaluation | SODA and GODAS | Jan 2000–Dec 2022 for each product |

Raw source catalogs are [CMIP6 via ESGF](https://pcmdi.llnl.gov/CMIP6/), [SODA](https://www.soda.umd.edu/), and [NOAA GODAS](https://psl.noaa.gov/data/gridded/data.godas.html). These catalogs do not serve the derived experiment packets. Place the original, trusted `default_server.pt`, `raw_windows.pt`, and `bundle.pt` in `source/remist_confirmatory_20261107/inputs/`. The assembler checks their SHA-256 values and preserves the original cohort and training-only statistics. The raw-to-packet reconstruction and distribution of these files and the fitted interface are separate from this code release.

## Run on a CUDA host with the required inputs

From the repository root:

```bash
cd source/remist_confirmatory_20261107
python code/prepare.py
python code/train.py --kind remist --seed 17 --budget 600 --multiplier 2 --stage final
python ../../tools/score_checkpoint.py jobs/final/remist_17_600_lr2/checkpoint.pt
```

`prepare.py` writes separate training/validation and evaluation packets. Fitting reads only the former. The fitting script validates every 50 updates and retains the checkpoint with minimum GFDL index MSE. Repeat for seeds 17, 29, 43, 71 and 101 and budgets 600 and 1800. Paper MSE is the mean across individually scored seeds in °C². Mean ACC averages fixed-lead Pearson correlations across leads and seeds.

For the whole-module comparison, continue from the same directory after preparing the packets:

```bash
cd module_ablation_20261108
mkdir -p data
ln -s ../../data/train.pt data/train.pt
ln -s ../../data/receipt.json data/receipt.json
python code/train.py --kind without_stir --seed 17 --budget 600 --multiplier 0.5 --stage final
python ../../../tools/score_checkpoint.py --ablation jobs/final/without_stir_17_600_lr0.5/checkpoint.pt
```

Use the separate ablation protocol for `full`, `without_vmir`, and `without_rcer`. The ablation `full` is the paired reference for those runs.

## Verification

On the original A6000 environment, the portable training entry point completed a two-update ReMiST smoke run. Replaying the archived seed-17, 600-update ReMiST and without-STIR checkpoints produced exactly the archived predictions on all 276 issue months per product for both SODA and GODAS. The replay used the original prepared packets and initializer.

## Licensing

The repository's MIT license covers the original code. The vendored Mamba3 file retains its copyright header and Apache-2.0 license. Upstream data and DPOT assets are governed by their providers' terms.
