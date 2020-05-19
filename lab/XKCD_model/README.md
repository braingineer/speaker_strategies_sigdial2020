# XKCD Model

- **Description**
    - The XKCD model is a reference model common to all speaker models in the paper
    - There are 3 steps to training the model: 
        1. training an uncalibrated version
        2. making the calibrations
        3. training a calibrated version
- configs
    - The YAML configuration files for the 3 steps (prefixe E001, E002, and E003)
- logs
    - The output of the 3 steps is stored here
- scripts
    - the python files for executing the 3 steps are here:
        1. train_uncalibrated_model.py
        2. make_calibrations.py
        3. train_calibrated_model.py
    - See `lab/README.md` for how they should be run