# Model Training and Analysis 

In this directory, `lab`, we have organized the model training and analysis code. 

To reproduce the results in the paper:

0. Install the magis_sigdial2020 and pyromancy libraries
    ```bash
    # From repository root
    # TODO: finish requirements.txt for reproducible environment
    cd src
    python setup.py install
    ```
1. Train the reference model using the scripts in `XKCD_model/scripts`
    ```bash
    # From repository root
    cd lab/XKCD_model/scripts
    python train_uncalibrated_model.py ../configs/E001_XKCDModel_uncalibrated.yaml
    python make_calibrations.py ../configs/E002_calibration_data.yaml
    python train_calibrated_model.py ../configs/E003_XKCDModel_calibrated.yaml
    ```
2. Evaluate the speaker reasoning models using the scripts in `analyses/scripts`
    ```bash
    # from repository root
    cd lab/analyses/scripts
    python evaluate_models_on_cic.py ../configs/E004_evaluate_on_cic.yaml
    ```
3. Run the 2 notebooks in `analyses/notebooks`

Note: some minor variations might exist from the exact numbers in the paper. 
Random numbers were controlled for as best as possible
(see `magis_sigdial2020.trainers.base_trainer.BaseTrainer.reset` for seed control), 
sometimes it's not possible to be 100% reproducible 
(see [PyTorch documentation on reproducbility](https://pytorch.org/docs/stable/notes/randomness.html))

