import os

from magis_sigdial2020.utils.plot import plot_row
from magis_sigdial2020.utils.data import Context
from magis_sigdial2020.datasets.xkcd import XKCD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as scistats

def get_backoff_p_w(xkcd=None):
    if xkcd is None:
        xkcd = XKCD.from_settings(coordinate_system="fft")
        
    backoff_p_w = np.zeros(829)
    for _, label_index in xkcd.train_fast:
        backoff_p_w[label_index] += 1
    backoff_p_w /= backoff_p_w.sum()
    
    return backoff_p_w

def aggregate_results(df, grouping_cols):
    agg_set = [
        ('perplexity', lambda x: np.exp(-1 * np.mean(x))),
        ('NLL_mean', lambda x: -1 * np.mean(x)),
        ('NLL_sd', lambda x: np.std(x)),
        ('NLL_min', lambda x: -1 * np.min(x)),
        ('NLL_max', lambda x: -1 * np.max(x)),
        ('NLL_median', lambda x: -1 * np.median(x)),
        ('NLL_sum', lambda x: -1 * np.sum(x)),
        ('support', len)
    ]
    
    aggdf = (
        df
        .groupby(grouping_cols)
        .agg({'log_S0': agg_set, 'log_S1': agg_set, 'log_S2': agg_set})
        .reset_index()
    )
    n = len(grouping_cols)
    new_columns = list(aggdf.columns.get_level_values(0)[:n])
    new_columns += [f'{name0}_{name1}' for name0, name1 in zip(aggdf.columns.get_level_values(0)[n:], aggdf.columns.get_level_values(1)[n:])]
    aggdf.columns = new_columns
    return aggdf


def compute_best_lambda(df_subset, backoff_p_w, target_proba='s0', plot=True, ylim=None, xlim=None):
    proba = df_subset[target_proba].values
    pw = backoff_p_w[df_subset.utterance_indices.values]
    
    pw = pw[(~np.isnan(proba)), None]
    proba = proba[(~np.isnan(proba)), None]
    lambdas = np.linspace(0.001, 1-1e-5, 1000)[None, :]

    backoff_matrix = np.log(proba * lambdas + (1 - lambdas) * pw)
    nll_vector = backoff_matrix.mean(axis=0)

    best_lambda = lambdas.flatten()[nll_vector.argsort()[-1]]
    if plot:
        #plt.figure()
        plt.plot(lambdas.flatten(), nll_vector, label=df_subset.model_name.unique()[0]+f".{target_proba}", lw=3)
        plt.scatter([best_lambda], [nll_vector.max()], color='red', marker='x', s=100)
        if ylim is not None:
            plt.ylim(*ylim)
        if xlim is not None:
            plt.xlim(*xlim)
    return best_lambda

def compute_lambda_and_add_log_proba(df, backoff_p_w, target_probas=('S0', 'S1', 'S2'), best_lambdas=None, plot=True):
    if best_lambdas is None:
        best_lambdas = {}
        for target_proba in target_probas:
            best_lambdas[target_proba] = compute_best_lambda(df, backoff_p_w, target_proba=target_proba, plot=plot)
    else:
        if not isinstance(best_lambdas, dict):
            raise Exception(f"Need mapping from target_probas ({target_probas}) to best_lambdas; "
                            f"Found {type(best_lambdas)} instead")
        if len(set(best_lambdas.keys()).intersection(set(target_probas))) != len(target_probas):
            raise Exception(f"Need mapping from target_probas ({target_probas}) to best_lambdas; "
                            f"Keys in best_lambdas: {best_lambdas.keys()}; target_probas: {target_probas}")
    for target_proba in target_probas:
        best_lambda = best_lambdas[target_proba]
        proba = df[target_proba].values
        pw = backoff_p_w[df.utterance_indices.values]
        df[f'{target_proba}_adjusted'] = proba * best_lambda + (1 - best_lambda) * pw
        df[f'log_{target_proba}'] = df[f'{target_proba}_adjusted'].apply(np.log)
        df[f'{target_proba}_best_lambda'] = best_lambda
    return df, best_lambdas
        
def load_results(path, backoff_p_w, verbose=False, grouping_keys=['model_name', 'model_class', 'algorithm_class', 'uses_ooc']):
    raw_results_df = pd.read_csv(path)
    processed_results_df = []
    
    for _, subdf in  raw_results_df.groupby(grouping_keys):
        train_subdf, best_lambdas = compute_lambda_and_add_log_proba(
            subdf[subdf.split=='train'].copy(), 
            backoff_p_w, 
            plot=verbose
        )
        dev_subdf, _ = compute_lambda_and_add_log_proba(
            subdf[subdf.split=='dev'].copy(), 
            backoff_p_w, 
            best_lambdas=best_lambdas,
            plot=verbose
        )
        test_subdf, _ = compute_lambda_and_add_log_proba(
            subdf[subdf.split=='test'].copy(), 
            backoff_p_w, 
            best_lambdas=best_lambdas,
            plot=verbose
        )
        processed_results_df.extend([train_subdf, dev_subdf, test_subdf])
    return pd.concat(processed_results_df)
    
    
def get_sorted_results(results_df, grouping_keys_for_loading, split):
    grouping_cols_for_agg = grouping_keys_for_loading + ['split']
    
    results_agg_df = aggregate_results(results_df, grouping_cols_for_agg)
    results_agg_subset_df = results_agg_df[results_agg_df.split==split].copy()
    results_agg_subset_df = results_agg_subset_df.rename(columns={
        "log_S0_perplexity": "S0_perplexity",
        "log_S1_perplexity": "S1_perplexity",
        "log_S2_perplexity": "S2_perplexity",
        "log_S2_support": "support"
    })
    cols = ['model_name', 'S0_perplexity', 'S1_perplexity','S2_perplexity', 'support', 'top_perplexity']
    results_agg_subset_df['top_perplexity'] = results_agg_subset_df.apply(lambda row: min(row.S0_perplexity, row.S1_perplexity), axis=1)
    return results_agg_subset_df[cols].sort_values('top_perplexity')[cols[:-1]]

def make_longform_results_df(sorted_agg_df, mapping_key={"CB": ("S1",), "RSA": ("S0", "S1",), "RGC": ("S0",)}):
    long_results_df = []
    for model in sorted_agg_df.model_name.unique():
        subset_df = sorted_agg_df[sorted_agg_df.model_name==model]
        # model encoded key with dashes
        model_key = model.split("-")[0]
        if model_key not in mapping_key:
            raise Exception("Assumption violated; expected model names to encode model type using a dash (e.g. X-Y, X is model type); "
                            f"Exception cause: model_name={model}; found model_key={model_key}; expected model_key to be one of {mapping_key.keys()}")
        for desired_probas in mapping_key[model_key]:
            new_df_i = pd.DataFrame({"perplexity": subset_df[f"{desired_probas}_perplexity"]})
            new_df_i["model_name"] = f"{model}-{desired_probas}"
            long_results_df.append(new_df_i)
    long_results_df = pd.concat(long_results_df)[["model_name", "perplexity"]]
    return long_results_df.sort_values("perplexity")    


class SignificanceTests:
    def __init__(self, rgc_probas, rsa_probas, cb_probas, cic, split):
        self.rgc_probas = rgc_probas
        self.rsa_probas = rsa_probas
        self.cb_probas = cb_probas
        self.cic = cic
        self.split = split
        self.analysis_df = self.make_analysis_df()
        
    def make_analysis_df(self):
        self.cic.set_split(self.split)
        self.cic.refresh_indices()
        df = []
        for i in range(self.rgc_probas.shape[0]):
            # for each split, the ith data point has the correct row index we need
            ith_datapoint = self.cic[i]
            row_dict = self.cic._df.iloc[ith_datapoint['row_index']].to_dict()
            
            rgc_proba = self.rgc_probas[i]
            rsa_proba = self.rsa_probas[i]
            cb_proba = self.cb_probas[i]
            row_dict["RGC"] = rgc_proba
            row_dict["RSA"] = rsa_proba
            row_dict["CB"] = cb_proba

            for epsilon in [0, 1e-3, 1e-2, 1e-1]:
                case_key = f"case_{epsilon}"
                if rgc_proba == 0 and rsa_proba == 0 and cb_proba == 0:
                    row_dict[case_key] = f"all == 0"
                elif rgc_proba == 0:
                    row_dict[case_key] = f"RGC == 0"
                elif rsa_proba == 0:
                    row_dict[case_key] = f"RSA == 0"
                elif cb_proba == 0:
                    row_dict[case_key] = "CB == 0"
                elif rsa_proba + epsilon < rgc_proba and cb_proba + epsilon < rgc_proba:
                    row_dict[case_key] = "RGC"
                elif rgc_proba + epsilon < rsa_proba and cb_proba + epsilon < rsa_proba:
                    row_dict[case_key] = "RSA"
                elif rgc_proba + epsilon < cb_proba and rsa_proba + epsilon < cb_proba:
                    row_dict[case_key] = "CB"
                else:
                    row_dict[case_key] = f"---"
            df.append(row_dict)
        return pd.DataFrame(df)
    
    def get_summary_dfs(self):
        return {
            "model_only": (
                self.analysis_df
                .groupby(['case_0'])
                .agg({'matcher_succeeded': ('mean', 'count')})
            ),
            "condition_only": (
                self.analysis_df
                .groupby(['condition'])
                .agg({'matcher_succeeded': ('mean', 'count')})
                .reindex(index=['close', 'split', 'far'])
            ),
            "condition_model": (
                self.analysis_df
                .groupby(['condition', 'case_0'])
                .agg({'matcher_succeeded': ('mean', 'count')})
                .reindex(index=['close', 'split', 'far'], level=0)
            )
        }
    
    def show_signifcance_utterance_probabilities_per_condition_across_models(self):
        conditions = ['far', 'split', 'close']
        models = ['RGC', 'RSA', 'CB']

        print(f"at 0.01 w/ bonferonni correction of 9 comparisons: {0.01/9:0.2e}")
        print(f"at 0.001 w/ bonferonni correction of 9 comparisons: {0.001/9:0.2e}")
        print(f"at 0.0001 w/ bonferonni correction of 9 comparisons: {0.0001/9:0.2e}")
        print("="*50)
        for condition in conditions:
            print("Condition: ", condition)
            stat_data = []
            for model in models:
                stat_data.append(
                    self.analysis_df.loc[
                        self.analysis_df.condition==condition,
                        model
                    ].values
                )

            print("Sizes: ", [stat_data_i.shape for stat_data_i in stat_data])
            print("Totals: ", [stat_data_i.sum() for stat_data_i in stat_data])

            for i, (model_i, stat_data_i) in enumerate(zip(models, stat_data)):
                for model_j, stat_data_j in zip(models[i+1:], stat_data[i+1:]):
                    for alternative in ["greater", "less"]:
                        outcome = scistats.wilcoxon(stat_data_i, stat_data_j, alternative=alternative)
                        sig = ""
                        if outcome.pvalue < 0.01 / 9:
                            sig += "*"
                        if outcome.pvalue < 0.001 / 9:
                            sig += "*"
                        if outcome.pvalue < 0.0001 / 9:
                            sig += "*"
                        print(f"{sig}{model_i} {alternative} than {model_j}")
                        print(f"\t{outcome}")
            print("-"*100)

    def show_signicance_matcher_success_per_model_across_conditions(self):
        conditions = ['far', 'split', 'close']
        models = ['RGC', 'RSA', 'CB']

        print(f"at 0.01 w/ bonferonni correction of 9 comparisons: {0.01/9:0.2e}")
        print(f"at 0.001 w/ bonferonni correction of 9 comparisons: {0.001/9:0.2e}")
        print(f"at 0.0001 w/ bonferonni correction of 9 comparisons: {0.0001/9:0.2e}")
        print("="*50)
        for model in models:
            print("Model: ", model)
            stat_data = []
            for condition in conditions:
                stat_data.append(
                    self.analysis_df[
                        (self.analysis_df.condition==condition) &
                        (self.analysis_df.case_0==model)
                    ]
                    .matcher_succeeded.values.astype(np.int32)
                )
            
            print("Sizes: ", [stat_data_i.shape for stat_data_i in stat_data])
            print("Totals: ", [stat_data_i.sum() for stat_data_i in stat_data])

            for i, (condition_i, stat_data_i) in enumerate(zip(conditions, stat_data)):
                for condition_j, stat_data_j in zip(conditions[i+1:], stat_data[i+1:]):
                    outcome = scistats.mannwhitneyu(stat_data_i, stat_data_j)
                    sig = ""
                    if outcome.pvalue < 0.01 / 9:
                        sig += "*"
                    if outcome.pvalue < 0.001 / 9:
                        sig += "*"
                    if outcome.pvalue < 0.0001 / 9:
                        sig += "*"
                    print(f"{sig}{condition_i} vs {condition_j}")
                    print(f"\t{outcome}")
            print("-"*100)
            
    def show_signicance_matcher_success_per_condition_across_models(self):
        conditions = ['far', 'split', 'close']
        models = ['RGC', 'RSA', 'CB']

        print(f"at 0.01 w/ bonferonni correction of 9 comparisons: {0.01/9:0.2e}")
        print(f"at 0.001 w/ bonferonni correction of 9 comparisons: {0.001/9:0.2e}")
        print(f"at 0.0001 w/ bonferonni correction of 9 comparisons: {0.0001/9:0.2e}")
        print("="*50)
        for condition in conditions:
            print("Condition: ", condition)
            stat_data = []
            for model in models:
                stat_data.append(
                    self.analysis_df[
                        (self.analysis_df.condition==condition) &
                        (self.analysis_df.case_0==model)
                    ]
                    .matcher_succeeded.values.astype(np.int32)
                )

            print("Sizes: ", [stat_data_i.shape for stat_data_i in stat_data])
            print("Totals: ", [stat_data_i.sum() for stat_data_i in stat_data])

            for i, (model_i, stat_data_i) in enumerate(zip(models, stat_data)):
                for model_j, stat_data_j in zip(models[i+1:], stat_data[i+1:]):
                    outcome = scistats.mannwhitneyu(stat_data_i, stat_data_j)
                    sig = ""
                    if outcome.pvalue < 0.01 / 9:
                        sig += "*"
                    if outcome.pvalue < 0.001 / 9:
                        sig += "*"
                    if outcome.pvalue < 0.0001 / 9:
                        sig += "*"
                    print(f"{sig}{model_i} vs {model_j}")
                    print(f"\t{outcome}")
            print("-"*100)
            
            
    def show_signicance_matcher_success_per_condition_vs_rgc0(self):
        conditions = ['far', 'split', 'close']
        models = ['RGC', 'RSA', 'CB']
        rgc0 = "RGC == 0"

        print(f"at 0.01 w/ bonferonni correction of 9 comparisons: {0.01/9:0.2e}")
        print(f"at 0.001 w/ bonferonni correction of 9 comparisons: {0.001/9:0.2e}")
        print(f"at 0.0001 w/ bonferonni correction of 9 comparisons: {0.0001/9:0.2e}")
        print("="*50)
        for condition in conditions:
            print("Condition: ", condition)
            stat_data = []
            for model in models:
                stat_data.append(
                    self.analysis_df[
                        (self.analysis_df.condition==condition) &
                        (self.analysis_df.case_0==model)
                    ]
                    .matcher_succeeded.values.astype(np.int32)
                )
            rgc0_stat_data = (
                self.analysis_df[
                    (self.analysis_df.condition==condition) &
                    (self.analysis_df.case_0==rgc0)
                ]
                .matcher_succeeded.values.astype(np.int32)
            )

            print("Sizes: ", [stat_data_i.shape for stat_data_i in stat_data])
            print("Totals: ", [stat_data_i.sum() for stat_data_i in stat_data])

            for i, (model_i, stat_data_i) in enumerate(zip(models, stat_data)):
                    outcome = scistats.mannwhitneyu(stat_data_i, rgc0_stat_data)
                    sig = ""
                    if outcome.pvalue < 0.01 / 9:
                        sig += "*"
                    if outcome.pvalue < 0.001 / 9:
                        sig += "*"
                    if outcome.pvalue < 0.0001 / 9:
                        sig += "*"
                    print(f"{sig}{model_i} vs RGC==0")
                    print(f"\t{outcome}")
            print("-"*100)