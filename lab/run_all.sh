#!/bin/sh

# 1. Train the reference model using the scripts in `XKCD_model/scripts`
cd XKCD_model/scripts
python train_uncalibrated_model.py ../configs/E001_XKCDModel_uncalibrated.yaml
python make_calibrations.py ../configs/E002_calibration_data.yaml
python train_calibrated_model.py ../configs/E003_XKCDModel_calibrated.yaml

# back to lab directory
cd ../..

# 2. Evaluate the speaker reasoning models using the scripts in `analyses/scripts`
cd analyses/scripts
python evaluate_models_on_cic.py ../configs/E004_evaluate_on_cic.yaml