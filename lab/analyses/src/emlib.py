import pandas as pd
import numpy as np

def get_emdata(df, probas_column, model_name_column='model_name', include_baselines=True, 
               backoff_p_w=None, annotate_names=True, utterance_index_column='utterance_indices'):
    em_data = []
    component_names = []
    utterance_index = None

    for model_name in df[model_name_column].unique():
        df_subset = df[df[model_name_column]==model_name].sort_values("row_indices")
        
        em_data.append(df_subset[probas_column].values)
        component_names.append(f'{model_name}_{probas_column}')
        utterance_index_i = df_subset[utterance_index_column].values
        if utterance_index is None:
            utterance_index = utterance_index_i
        else:
            assert np.allclose(utterance_index, utterance_index_i)
        
    if include_baselines:
        if backoff_p_w is None:
            raise Exception("To use baseslines, send in backoff_p_w")

        component_names.append('xkcd_baseline')
        em_data.append(backoff_p_w[utterance_index])
        
        component_names.append('random')
        em_data.append(np.ones_like(utterance_index) / backoff_p_w.shape[0])
        
    em_data = np.stack(em_data).T
    return em_data, component_names, utterance_index

    
def e_step(data, prior):
    p_membership = data * prior[None, :]
    p_membership /= p_membership.sum(axis=1, keepdims=True)
    return p_membership


def m_step(p_membership):
    prior_component = p_membership.mean(axis=0)
    return prior_component


def run_em(data, prior, num_iterations=100, verbose=True, verbose_frequency=50, component_names=None, threshold=1e-6):
    p_membership = e_step(data, prior)
    converged=False
    for i in range(num_iterations):
        p_membership_tm1 = p_membership
        prior_tm1 = prior
        
        p_membership = e_step(data, prior)
        prior = m_step(p_membership)
        update_difference = np.abs(prior_tm1 - prior)
        if update_difference.sum() < threshold:
            converged=True
            if verbose:
                print(f"Threshold reached; {update_difference.sum()}")
            break
        if verbose and (i + 1) % verbose_frequency == 0:
            print(f"-- iteration {i} --")
            print(f'Delta mean: {update_difference.mean()}, Delta sum: {update_difference.sum()}')
            if component_names:
                print(dict(zip(component_names, prior)))
            else:
                print(prior)
    if not converged:
        print("DID NOT EXIT VIA CONVERGENCE")
    return p_membership, prior


def em_experiment_test(df, posterior_from_train, s0_model_list=[], s1_model_list=[], 
                       model_name_column='model_name', verbose=True, return_results=False, 
                       s0_column_name='s0', s1_column_name='s1', utterance_index_column='utterance_indices', 
                       backoff_p_w=None, include_baselines=True):
    em_data = []
    if len(s0_model_list) > 0:
        s0_data, _, utterance_index = get_emdata(
            df=df[df[model_name_column].isin(s0_model_list)], 
            probas_column=s0_column_name, 
            model_name_column=model_name_column,
            utterance_index_column=utterance_index_column,
            backoff_p_w=backoff_p_w,
            include_baselines=False
        )        
        em_data.append(s0_data)

    if len(s1_model_list) > 0:
        s1_data, _, utterance_index = get_emdata(
            df=df[df[model_name_column].isin(s1_model_list)], 
            probas_column=s1_column_name, 
            model_name_column=model_name_column,
            utterance_index_column=utterance_index_column,
            backoff_p_w=backoff_p_w,
            include_baselines=include_baselines
        )    
        em_data.append(s1_data)
        
    em_data = np.concatenate(em_data, axis=1)
    
    p_membership = e_step(em_data, posterior_from_train)
    
    # em_data.shape == (num_data, num_components)
    # posterior_from_train == (num_components,)
    # new data probability is \sum_component p(x|component) * p(component)
    new_data_p = np.sum(em_data * posterior_from_train[None, :], axis=1)
    perplexity = np.exp(-1 * np.log(new_data_p).mean())
        
    if verbose:
        print(f'\tPerplexity of interpolated data: {perplexity}')
        print('\t' + "="*68)
        
    if return_results:
        return {'p_membership': p_membership, 
                'interpolated_data': new_data_p,
                'em_data': em_data,
                'perplexity': perplexity,
                'utterance_index': utterance_index}

def em_experiment_train(df, s0_model_list=[], s1_model_list=[], model_name_column='model_name', 
                        verbose=True, return_results=False, s0_column_name='s0', s1_column_name='s1',
                        utterance_index_column='utterance_indices', backoff_p_w=None, include_baselines=True,
                        num_iterations=5000):
    em_data = []
    component_names = []
    row_index = None
    for model_name in s0_model_list + s1_model_list:
        row_index_i = df[df[model_name_column]==model_name].row_indices.values
        if row_index is None:
            row_index = row_index_i
        else:
            assert np.allclose(row_index, row_index_i)
    
    
    if len(s0_model_list) > 0:
        s0_data, s0_names, utterance_index = get_emdata(
            df=df[df[model_name_column].isin(s0_model_list)], 
            probas_column=s0_column_name, 
            model_name_column=model_name_column,
            utterance_index_column=utterance_index_column,
            backoff_p_w=backoff_p_w,
            include_baselines=False
        )        
        em_data.append(s0_data)
        component_names.extend(s0_names)

    if len(s1_model_list) > 0:
        s1_data, s1_names, utterance_index = get_emdata(
            df=df[df[model_name_column].isin(s1_model_list)], 
            probas_column=s1_column_name, 
            model_name_column=model_name_column,
            utterance_index_column=utterance_index_column,
            backoff_p_w=backoff_p_w,
            include_baselines=include_baselines
        )    
        em_data.append(s1_data)
        component_names.extend(s1_names)
        
    em_data = np.concatenate(em_data, axis=1)

    p_membership, posterior = run_em(data=em_data, 
                                     prior=np.ones(len(component_names))/len(component_names), 
                                     verbose=False,
                                     num_iterations=num_iterations, 
                                     verbose_frequency=1000, 
                                     component_names=component_names)
    if verbose:
        print('\t' + "="*30 + "Evidence" + "="*30)
    new_data_p = np.zeros(em_data.shape[0])
    for i, (component_name, p_model) in enumerate(zip(component_names, posterior)):
        if verbose:
            print(f"\tEM Posterior for {component_name}: {p_model:0.6f}")
        new_data_p += em_data[:,i] * p_model
        
    perplexity = np.exp(-1 * np.log(new_data_p).mean())
    if verbose:
        print(f'\tPerplexity of interpolated data: {perplexity}')
        print('\t' + "="*68)
    
    if return_results:
        return {'p_membership': p_membership, 
                'posterior': posterior, 
                'interpolated_data': new_data_p,
                'model_names': component_names,
                'row_indices': row_index,
                'perplexity': perplexity,
                'utterance_index': utterance_index,
                'em_data': em_data}
    
def em_experiment(df, s0_model_list=[], s1_model_list=[], model_name_column='model_name', 
                  verbose=True, s0_column_name='s0', s1_column_name='s1',
                  utterance_index_column='utterance_indices', backoff_p_w=None, include_baselines=True,
                  num_iterations=5000):
    assert 'train' in df.split.unique()
    test_splits = []
    for possible_test_split in ['dev', 'val', 'test']:
        if possible_test_split in df.split.unique():
            test_splits.append(possible_test_split)
    
    train_results = em_experiment_train(
        df=df[df.split=='train'], s0_model_list=s0_model_list, s1_model_list=s1_model_list, 
        model_name_column=model_name_column, s0_column_name=s0_column_name, 
        s1_column_name=s1_column_name, backoff_p_w=backoff_p_w, return_results=True, verbose=False,
        include_baselines=include_baselines,
        num_iterations=num_iterations
    )
    results = {
        'posterior': train_results.pop('posterior'),
        'model_names': train_results.pop('model_names'),
        'row_indices': train_results.pop('row_indices'),
        'train': train_results
    }
    for test_split in test_splits:
        results[test_split] = em_experiment_test(
            df=df[df.split==test_split], posterior_from_train=results['posterior'], s0_model_list=s0_model_list, 
            s1_model_list=s1_model_list, model_name_column=model_name_column, s0_column_name=s0_column_name, 
            s1_column_name=s1_column_name, backoff_p_w=backoff_p_w, return_results=True, verbose=False,
            include_baselines=include_baselines
        )
    return results
    

def show_em_results(results_dict, splits_to_show=("train", "dev")):
    print("-------------------------- results -----------------------")
    for i, (model_name, p_model) in enumerate(zip(results_dict['model_names'], results_dict['posterior'])):
        print(f"\tEM Posterior for {model_name}: {p_model:0.6f}")
    for split in splits_to_show:
        print(f"\tPerplexity on {split}: {results_dict[split]['perplexity']}")
    print("----------------------------------------------------------")    