# Analyzing Speaker Strategy in Referential Communication

## Contents

This is the repository to accompany the paper:
```
Brian McMahan and Matthew Stone. 2020. Analyzing Speaker Strategy in 
Referential Communication.  In Proceedings of the SIGDIAL 2020 Conference.
```

#### Abstract

We analyze a corpus of referential communication through the lens of quantitative 
models of speaker reasoning.  Different models place different emphases on linguistic 
reasoning and collaborative reasoning.  This leads models to make different assessments 
of the risks and rewards of using specific utterances in specific contexts.  By 
fitting a latent variable model to the corpus, we can exhibit utterances that 
give systematic evidence of the diverse kinds of reasoning speakers employ, and 
build integrated models that recognize not only speaker reference but also speaker 
reasoning. 
    
## Documentation

### Repository contents

- data/
    + filteredCorpus.csv
        - The Colors in Context (CIC) dataset. ([ref 1](https://cocolab.stanford.edu/datasets/colors.html), [ref 2](https://www.aclweb.org/anthology/Q17-1023.pdf))
    + cic_vectorized.csv
        - A parsed & vectorized form of the CIC dataset
        - Made with [magis_sigdial2020.datasets.color_reference_2017.vectorized.make_or_load_cic()](https://github.com/braingineer/speaker_strategies_sigdial2020/blob/master/src/magis_sigdial2020/datasets/color_reference_2017/vectorized.py))
    + xkcd/
        - the XKCD dataset ([ref 1](https://blog.xkcd.com/2010/05/03/color-survey-results/), [ref 2](https://www.aclweb.org/anthology/Q15-1008.pdf))
        - annotations.csv
            - descriptions of color patches
        - color_values.npy
            - color patch values
        - vocab.json
            - cached XKCD lexicon (sorted lexicographically, so it is reproducible, but this is easier than re-loading annotatins.csv if you just want the vocab)
- lab/
    + **Description**
        - The `lab` directory is where model training and analyses are housed
    + README.md
        - details around the reproducibility are here
    + run_all.sh
        - A script which will run most of the steps for reproducibility
    + XKCD_model/
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
    + analyses/
        - **Description**
            - The reference model is used in 3 kinds of speaker models 
            - The speaker models are evaluated on the Colors in Context dataset
            - The analysis numbers in the paper come from the notebooks in this directory
        - configs
            - The YAML configuration file for running the evaluation
        - notebooks
            - Two analyses:
                1. A001: compare speaker models independently
                    - The numbers for Section 4.2
                    - reports perplexity on CIC splits and the statistics around that
                2. A002: compare speaker models as a mixture
                    - The numbers for Section 4.3
                    - fits a mixture model to speaker model predictions
                    - reports winning model, matcher successes, and statistics around that   
        - scripts
            - evaluate_models_on_cic.py
                - The python script for loading the speaker models, evaluating on CIC, and storing the result
        - src
            - cic_results_lib.py
                - Results loading and statistical tests
            - emlib.py
                - EM routine for the mixture analysis
- src/
    + sigdial_20202/
        - contains the model, model training, and dataset code
    + pyromancy/
        - used for ML logging
    + requirements.txt
    + setup.py
- tests
    
### Reproducibility

See [`lab/README.md`](lab/README.md) and [`lab/run_all.sh`](lab/run_all.sh)

### Small note...

This repo has the data in it. As it stands, the total repo size is around 200MB.