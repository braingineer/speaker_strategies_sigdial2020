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

### Libraries

Two libraries in this repository:
- sigdial_20202
    - contains the model, model training, and dataset code
- pyromancy
    - used for ML logging
    
### Reproducibility

See [`lab/README.md`](lab/README.md) and [`lab/run_all.sh`](lab/run_all.sh)

### Small note...

This repo has the data in it. As it stands, the total repo size is around 200MB.