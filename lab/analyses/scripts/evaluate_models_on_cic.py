import argparse
import json
import os
import pathlib
import yaml
import gc
import time

from magis_sigdial2020.hyper_params import HyperParameters
from magis_sigdial2020.datasets.color_reference_2017.vectorized import make_or_load_cic
from magis_sigdial2020.models.xkcd_model import XKCDModelWithRGC
from magis_sigdial2020.utils.data import Context
from magis_sigdial2020.algorithms import rgc, rsa_ooc, rsa_mg
from magis_sigdial2020 import settings
import pyromancy
import pyromancy.reader as reader
import pyromancy.subscribers as sub
import numpy as np
import pandas as pd
import torch
import tqdm


RUNTIME_INFO = {
    "LAB_SUBDIR_ROOT": pathlib.Path(__file__).absolute().parents[1],
    "LAB_ROOT": f"{settings.REPO_ROOT}/lab"
}

MODEL_CLASSES = {
    'XKCDModelWithRGC': XKCDModelWithRGC,
}

ALGORITHM_CLASSES = {
    'RGC.core': rgc.RGCAlgorithm,
    'RSA.OOC': rsa_ooc.LockedAlpha,
    'RSA.MG': rsa_mg.LockedAlpha,
}

        
def instantiate_models(hparams):
    """Instantiate the models using the YAML config
    
    Note:
        In this version, I am evaluating models from pyromancy output folders
        Specifically, the folder specified by `experiment_name`
    
    YAML format:
        
        # experiment settings
        experiment_name: E004_evaluate_on_xkcd
        trial_name: published_version
        root_output_path: "{LAB_SUBDIR_ROOT}/logs"
        device: cuda

        model_variations:
            - model_name: 
              algorithm_class: 
              algorithm_kwargs:
                kwarg1: value1
                ...
              reference_model:
                dir: 
                class:
        
    """
    model_dicts = []
    pyromancy.settings.set_root_output_path(hparams.root_output_path)
    hparams.all_algorithm_kwargs = set()
    for model_info in hparams.models:
        hparams.all_algorithm_kwargs.update(set(model_info['algorithm_kwargs'].keys()))
        
        reference_model_dict = model_info["reference_model"]
        hparams_yaml = f"{reference_model_dict['dir']}/hparams.yaml"
        reference_model_hparams = HyperParameters.load(hparams_yaml)
        reference_model_hparams.trial_path = reference_model_dict['dir']
        ReferenceModelClass = MODEL_CLASSES[reference_model_dict['class']]
        reference_model = ReferenceModelClass.make(
            reference_model_hparams, reload=True, eval_mode=True
        )
        
        AlgorithmClass = ALGORITHM_CLASSES[model_info['algorithm_class']]
        model_info["speaker_model"] = AlgorithmClass(
            reference_model, 
            **model_info['algorithm_kwargs']
        )
        
        model_dicts.append(model_info)
    
    hparams.all_algorithm_kwargs = list(hparams.all_algorithm_kwargs)
    return model_dicts
        

def parse_hparams():
    """Input a YAML file 
    
    YAML format:
        
        # experiment settings
        experiment_name: E004_evaluate_on_xkcd
        trial_name: published_version
        root_output_path: "{LAB_SUBDIR_ROOT}/logs"
        device: cuda

        model_variations:
            - model_name: 
              algorithm_class: 
              algorithm_kwargs:
                kwarg1: value1
                ...
              reference_model:
                dir: 
                class: 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("YAML_CONFIG", help='')
    hparams = HyperParameters.load(parser.parse_args().YAML_CONFIG, RUNTIME_INFO)
    return hparams


def run_eval_loop(model_dict, dataset, hparams):
    """Evaluate 1 model on 1 split
    
    Iterate over batches, aggregate outputs.
    """
    output_df = {
        "row_indices": [], 
        "utterance_indices": [],
        "SEM": [], 
        "S0": [], 
        "L0": [],
        "S1": [],
        "L1": [],
        "S2": [],
        "L2": []
    }
    to_numpy = lambda x: x.cpu().detach().numpy()
    
    model = model_dict['speaker_model']
    model.model = model.model.to(hparams.device)
    batch_size = getattr(hparams, "batch_size", 256)
    
    batch_bar = tqdm.tqdm(total=len(dataset)//batch_size, leave=False, position=1)
    batch_generator = dataset.generate_batches(
        batch_size=batch_size, 
        device=hparams.device, 
        drop_last=False, 
        shuffle=False
    )
    
    for batch in batch_generator:
        
        utterance_indices = batch['y_utterance'].view(-1,1)
        row_indices = batch['row_index']
        batch_size_i = utterance_indices.shape[0]
        
        output_df['row_indices'].append(to_numpy(row_indices))
        output_df['utterance_indices'].append(to_numpy(utterance_indices.view(-1)))
        
        context = model(Context.from_cic_batch(batch))
        
        output_df["SEM"].append(to_numpy(
            context.SEM_probabilities[:, 0]
            .gather(dim=1, index=utterance_indices)
            .view(-1)
        ))
        
        for level in [0, 1, 2]:
            S_short = f"S{level}"
            S_long = f"S{level}_probabilities"
            L_short = f"L{level}"
            L_long = f"L{level}_probabilities"
            
            if hasattr(context, S_long):
                output_df[S_short].append(to_numpy(
                    getattr(context, S_long)[:, 0]
                    .gather(dim=1, index=utterance_indices)
                    .view(-1)
                ))
                output_df[L_short].append(to_numpy(
                    getattr(context, L_long)[:, 0]
                    .view(-1)
                ))
            else:
                dummy_vector = np.ones((batch_size_i,)).astype(np.float32) * -1
                output_df[S_short].append(dummy_vector)
                output_df[L_short].append(dummy_vector)
                
        batch_bar.update()
    
    output_df = {name: np.concatenate(vector_list) for name, vector_list in output_df.items()}
    output_df = pd.DataFrame(output_df)
    
    output_df["model_name"] = model_dict["model_name"]
    output_df["algorithm_class"] = model_dict["algorithm_class"]
    output_df["reference_model_dir"] = model_dict["reference_model"]["dir"]
    output_df["reference_model_class"] = model_dict["reference_model"]["class"]
    # hparams.all_algorithm_kwargs is set in `instantiate_models`
    # this helps with no weird entries in the final dataframe, which concats all
    # individual dataframes (e.g. if algorithm_kwargs are different between models)
    for kwarg_key in hparams.all_algorithm_kwargs:
        output_df[kwarg_key] = model_dict['algorithm_kwargs'].get(kwarg_key, "---")
    
    del model
    del model_dict['speaker_model']
    del context
    time.sleep(0.1)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return output_df
    
    
def main():
    """ Entry point for the evaluation """
    hparams = parse_hparams()
    
    pyromancy.settings.set_root_output_path(hparams.root_output_path)
    exp = pyromancy.initialize(
        experiment_name=hparams.experiment_name,
        subscribers=[sub.DBSubscriber()],
        trial_name=hparams.trial_name
    )
    exp.log_exp_start()
    
    hparams.results_filepath = exp.expand_to_trial_path("results.csv")
    hparams.hparams_filepath = exp.expand_to_trial_path("hparams.yaml")
    hparams.save(hparams.hparams_filepath)
    
    cic = make_or_load_cic()
    cic.set_difficulty_subset([0])
    
    results_df = []
    for split in ['train', 'dev', 'test']:
        cic.set_split(split)
        cic.refresh_indices()
        
        # reinstantiating inside the outer loop 
        # I do some judicial garbage collecting / cache clearing to avoid some
        # weird OOM errors on GPUs. 
        # so they need to be re-instantiated each time.
        model_dicts = instantiate_models(hparams)
        
        for model_dict in tqdm.tqdm(model_dicts, desc=f'models on {split}', position=0):
            results_df_i = run_eval_loop(model_dict, cic, hparams)
            results_df_i['split'] = split
            results_df.append(results_df_i)

    results_df = pd.concat(results_df)
    results_df.to_csv(hparams.results_filepath, index=None)
    
    exp.log_exp_end()
    
if __name__ == "__main__":
    with torch.no_grad():
        main()