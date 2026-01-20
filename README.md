# Analysis of PINPOINT data
- `preprocessing.py` modifies data in `data` directory:
  - By default this should be outside of the analysis directory, path can be changed in `get_data_path` function in `utils.py`
  - Expects common ROOT files in `data/root` directory and will create new `parquet` and `torch` directories with preprocessed data
- `cnn.py` runs neutrino flavor identification using 2d CNNs with zx and zy projections of binned pixel hits. This is meant to be run on a remote machine with a powerful GPU
- `evaluate_cnn.ipynb` takes weights from `cnn.py` and is meant to be run locally to produce performance plots