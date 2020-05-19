from argparse import Namespace
import os
import sqlite3

import numpy as np
import pandas as pd
from pyromancy.settings import MESSAGES, CHANNELS


def maybe_makedir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def trial_finished(trial_path):
    dbpath = os.path.join(trial_path, 'events.db')
    conn = sqlite3.connect(dbpath)
    try:
        df = pd.read_sql(f"select * from {CHANNELS.TRAINING_EVENTS} where event_name='{MESSAGES.EXPERIMENT_END}'", conn)
    except pd.io.sql.DatabaseError:
        conn.close()
        return False
    conn.close()
    return len(df) == 1


def get_args(trial_path):
    dbpath = os.path.join(trial_path, 'events.db')
    conn = sqlite3.connect(dbpath)
    df = pd.read_sql(f"select * from {CHANNELS.HYPERPARAM_EVENTS}", conn)
    assert len(df) == 1
    arg_dict = df.iloc[0].to_dict()
    fixed_arg_dict = {}
    for key, value in arg_dict.items():
        if isinstance(value, np.int64):
            value = int(value)
        elif isinstance(value, np.float64):
            value = float(value)
        fixed_arg_dict[key] = value
    args = Namespace(**fixed_arg_dict)
    conn.close()
    return args


def get_specific_args(exp_name, trial_name):
    # TODO: see if this is necessary to avoid circular imports.. being paranoid for now
    from pyromancy.reader import SingleExperimentReader
    exp = SingleExperimentReader(exp_name)
    trial_map = {os.path.split(trial_path)[1]: trial_path for trial_path in exp.all_trial_paths}
    args = get_args(trial_map[trial_name])
    args.trial_path = trial_map[trial_name]
    return args


def infer_critical_hp(df, known_non_hp=set(['epoch', 'accuracy', 'loss', 'perplexity', 'split',
                                           'timestamp', 'trial_name', 'device', 'cuda', 'patience_threshold',
                                           'checkpoint_metric', 'num_epochs'])):
    crit_hp = []
    for colname in df.columns:
        if colname in known_non_hp:
            continue
        else:
            num_values = len(df[colname].unique())
            if num_values >= 2 and num_values <=20:
                crit_hp.append(colname)
    return crit_hp