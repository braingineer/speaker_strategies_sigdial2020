import logging
import sqlite3
import pandas as pd
import glob
import os
from pyromancy import settings, utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SingleExperimentReader:
    def __init__(self, experiment_name, filter_unfinished=True):
        self.experiment_path = os.path.join(settings.get_root_output_path(),
                                            experiment_name)
        self.all_trial_paths = glob.glob(os.path.join(self.experiment_path, "*"))
        if filter_unfinished:
            self.filter_unfinished()
        self._crit_df = None

    def filter_unfinished(self):
        self.all_trial_paths = list(filter(utils.trial_finished, self.all_trial_paths))

    def _make_critical_df(self):
        all_df = []
        for trial_path in self.all_trial_paths:
            db_path = os.path.join(trial_path, "events.db")
            if not os.path.exists(db_path):
                logger.info(f"{trial_path} does not have an events.db")
                continue

            try:
                conn = sqlite3.connect(db_path)

                metric_df = pd.read_sql("select * from metric_events", conn)
                hp_df = pd.read_sql("select * from hyperparam_events", conn)
                # training_df = pd.read_sql("select * from training_events", conn)

                hdf_col = set(hp_df.columns) - set(metric_df.columns) - set(['hp_search'])
                hdf_col.update(set(['experiment_name', 'trial_name']))
                all_df.append(metric_df.merge(hp_df[list(hdf_col)],
                                              on=['experiment_name', 'trial_name']))
            except pd.io.sql.DatabaseError as d:
                # TODO: convert to logger
                print(f"{trial_path} failed with database error: {d}")
            except Exception as e:
                # TODO: Convert to logger
                print(f"Exception ({type(e)}: {e}")

        return pd.concat(all_df)

    def get_critical_df(self):
        if self._crit_df is None:
            self._crit_df = self._make_critical_df()
        return self._crit_df

    def get_all_args(self):
        all_args = []
        for trial_path in self.all_trial_paths:
            try:
                args = utils.get_args(trial_path)
                args.trial_path = trial_path
                all_args.append(args)
            except AssertionError:
                print(f'{trial_path} failed; more than 1 hyper parameter event')
        return all_args


def ls(pattern="*"):
    exp_names = []
    for path in glob.glob(os.path.join(settings.get_root_output_path(), pattern)):
        if os.path.isdir(path):
            exp_names.append(os.path.split(path)[1])
    return exp_names
