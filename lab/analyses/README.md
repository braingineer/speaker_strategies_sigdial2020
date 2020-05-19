# Analyses

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