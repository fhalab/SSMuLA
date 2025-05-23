# SSMuLA

## About
* Code base termed "Site Saturation Mutagenesis Landscape Analysis (SSMuLA)" for our [paper](https://doi.org/10.1101/2024.10.24.619774) titled "Evaluation of Machine Learning-Assisted Directed Evolution Across Diverse Combinatorial Landscapes"
* Data and results can be found at [Zenodo](https://doi.org/10.5281/zenodo.13910506)
![fig1](fig1.png)

### Environment
* For the overall environment `SSMuLA`
```
conda env create -f SSMuLA.yml
```
* Then install EVmutation from the [develop branch](https://github.com/debbiemarkslab/EVcouplings/archive/develop.zip) after the environment is created
* For the ESM-IF environment
```
conda create -n inverse python=3.9
conda activate inverse
conda install pytorch cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg -c conda-forge
conda install pip
pip install biotite
pip install git+https://github.com/facebookresearch/esm.git
```
or install the ESM-IF environment `esmif`
```
conda env create -f esmif.yml
```
* For the CoVES environment `coves`
```
conda env create -f coves.yml
```
* For installing Triad command line, see instructions [here](https://triad.protabit.com/api/static/doc/user/userGettingStarted.html)
* For running ESM-2 fintuning simulations, use the `finetune.yml` environment
```
conda env create -f finetune.yml
```
* Frozen environment can be found in `envs/frozen`

### Datasets
* The `data/` folder is organized by protein type. Each protein directory contains:
    - `.fasta`: FASTA file for the parent sequence
    - `.pdb`: PDB file for the parent structure
    - `.model`: EVmutation model file
    - `fitness_landscape/`: Folder containing CSV files for all fitness landscapes for this protein type, each listing amino acid substitutions and their corresponding fitness values from the original sources
    - `scale2max/`: the folder containing processed fitness csv files returned from the `process_all` function in the `SSMuLA.fitness_process_vis` module where the maximum fitness value is normalized to 1 for each landscape

* Landscapes summarized in the table below and described in detail in the paper:

| Landscape | PDB ID | Sites                  |
|-----------|--------|------------------------|
| ParD2     | 6X0A   | I61, L64, K80           |
| ParD3     | 5CEG   | D61, K64, E80           |
| GB1       | 2GI9   | V39, D40, G41, V54      |
| DHFR      | 6XG5   | A26, D27, L28           |
| T7        | 1CEZ   | N748, R756, Q758        |
| TEV       | 1LVM   | T146, D148, H167, S170  |
| TrpB3A    | 8VHH   | A104, E105, T106        |
| TrpB3B    |        | E105, T106, G107        |
| TrpB3C    |        | T106, G107, A108        |
| TrpB3D    |        | T117, A118, A119        |
| TrpB3E    |        | F184, G185, S186        |
| TrpB3F    |        | L162, I166, Y301        |
| TrpB3G    |        | V227, S228, Y301        |
| TrpB3H    |        | S228, G230, S231        |
| TrpB3I    |        | Y182, V183, F184        |
| TrpB4     |        | V183, F184, V227, S228  |


### Preprocessing
* Run
```
python -m tests.test_preprocess
```
refer to the test file and the script documentation for further details
* Processed with `fitness_process_vis`
* Rename columns to be `AAs`, `AA1`, `AA2`, `AA3`, `AA4`, `fitness`, add `active` if not already there and add `muts` columns
* Scale to `max` (with option to scale to `parent`)
* Processed data saved in `scale2max` folder
* The landscape stats will be saved 


### Landscape attributes
#### Local optima
* Run
```
python -m tests.local_optima
```
refer to the test file and the script documentation for further details
* Calculate local optima with `calc_local_optima` function in `SSMuLA.local_optima`

#### Pairwise epistasis
* Run
```
python -m tests.pairwise_epistasis
```
refer to the test file and the script documentation for further details
* Calculate pairwise epistasis with `calc_all_pairwise_epistasis` function in `SSMuLA.pairwise_epistasis`
* Start from all active variants scaled to max fitness without post filtering
* Initial results will be saved under the default path `results/pairwise_epistasis` folder (corresponding to the `active_start` subfolder in the zenodo repo)
* Post processing the output with `plot_pairwise_epistasis` function in `SSMuLA.pairwise_epistasis`
* Post processed results will be saved under the default path `results/pairwise_epistasis_dets` folder with summary files (corresponding to the `processed` subfolder) and `results/pairwise_epistasis_vis` for each of the landscape with a master summary file across all landscapes (in the `pairwise_epistasis_summary.csv`)

### Zero-shot
* The currrent pipeline runs EVmutation and ESM together, and then append the rest based

#### EVmutation
* All EVmutation predictions run with [EVcouplings](https://v2.evcouplings.org/)
* All settings remain default
* Model parameters in the `.model` files are downloaded and renamed

#### ESM
* The logits will be generated and saved in the output folder
* Run
```
python -m tests.test_ev_esm
```
refer to the test file and the script documentation for further details

#### Hamming distance 
* Directly calculated from `n_mut`  
* For Hamming ditsance testing, run
```
python -m tests.hamming_distance
```
to deploy `run_hd_avg_fit` and `run_hd_avg_metric` from `SSMuLA.calc_hd`
refer to the test file and the script documentation for further details

#### ESM-IF
* Run
```
python -m tests.test_esmif
```
refer to the test file and the script documentation for further details
* Generate the input fasta files with `get_all_mutfasta` from `SSMuLA.zs_data` to be used in ESM-IF
* Set up the environment for [ESM-IF](https://github.com/facebookresearch/esm?tab=readme-ov-file#invf) to 
```
conda create -n inverse python=3.9
conda activate inverse
conda install pytorch cudatoolkit=11.3 -c pytorch
conda install pyg -c pyg -c conda-forge
conda install pip
pip install biotite
pip install git+https://github.com/facebookresearch/esm.git
```
or use 
* With in the `esmif` folder within the new environment, run
```
./esmif.sh
```
* ESM-IF results will be saved in the same directory as the `esmif.sh` script

#### CoVES
* Follow the instructions in the [CoVES](https://github.com/ddingding/CoVES/tree/publish)
* Prepare input data in the `coves_data` folder
* Run `run_all_coves` from `SSMuLA.run_coves` to get all scores
* Append scores with `append_all_coves_scores` from `SSMuLA.run_coves`

#### Triad
* Prep mutation file in `.mut` format such as `A_1A+A_2A+A_3A+A_4A` with `TriadGenMutFile` class in `SSMuLA.triad_prepost` 
* Run
```
python -m tests.test_triad_pre
```
refer to the test file and the script documentation for further details
* With `triad-2.1.3` local command line
* Prepare structure with `2prep_structures.sh`
* Run `3getfixed.sh`
* Parse results with `ParseTriadResults` class in `SSMuLA.triad_prepost`


#### Combine all zs
* Run
```
python -m tests.test_zs
```
refer to the test file and the script documentation for further details

### Simulations
#### DE
* Run `de_simulations` and visualise with `plot_de_simulations`
* Run
```
python -m tests.test_de
```
and
```
python -m tests.test_de_vis
```
refer to the test file and the script documentation for further details


#### MLDE and ftMLDE
* Use `MLDE_lite` environment
* For using learned ESM embeddings, first run `gen_all_learned_emb` from `SSMuLA.gen_learned_emb`, else skip this step
* Run
```
python -m tests.test_gen_learned_emb
``` 
* Run `run_all_mlde_parallelized` from `SSMuLA.mlde_lite` to run simulations
* Run
```
python -m tests.test_mlde
```
* Important options including:
    * `n_mut_cutoffs`: list of integers for Hamming distance cutoff options where `[0]` means none and `[2]` for Hamming distance of two for ensemble
    * `zs_predictors`: list of strings for zero-shot predictors, i.e. `["none", "Triad", "ev", "esm"]` where `none` means not focused training and thus default MLDE runs; the list can be extended for non-Hamming distance ensemble, including `["Triad-esmif", "Triad-ev", "Triad-esm", "two-best"]`
    * `ft_lib_fracs`: list of floats for fraction of libraries to use for focused training, i.e. `[0.5, 0.25, 0.125]`
    * `encoding`: list of strings for encoding options, i.e. `["one-hot"] + DEFAULT_LEARNED_EMB_COMBO`
    * `model_classes`: list of strings for model classes, i.e. `["boosting", "ridge"]`
    * `n_samples`: list of integers for number of training samples to use, i.e. `[96, 384]`
    * `n_split`: integer for number of splits for cross-validation, i.e. `5`
    * `n_replicate`: integer for number of replicates for each model, i.e. `50`
    * `n_tops`: integer for number of variants to test the prediction, i.e. `[96, 384]`
refer to the test file and the script documentation for further details
* Run `MLDESum` from `SSMuLA.mlde_analysis` to get the summary dataframe and optional visualization
```
python -m tests.test_mlde_vis
```

#### ALDE and ftALDE
* See details in [alde4ssmula](https://github.com/fhalab/alde4ssmula) repository
* `aggregate_alde_df` from `SSMuLA.alde_analysis` to get the summary dataframe
```
python -m tests.test_alde
```

#### Fine-tuning
* Run `train_predict_per_protein` from `SSMuLA.plm_finetune` for ESM-2 LoRA fine-tuning simulations

### Analysis and paper figures
* All notebooks in `fig_notebooks` are used to reproduce figures in the paper with files downloaded from [Zenodo]((https://doi.org/10.5281/zenodo.13910506))

## Contact
* [Francesca-Zhoufan Li](mailto:fzl@caltech.edu)