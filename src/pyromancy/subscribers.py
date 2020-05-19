"""
The subscribers listen to compute messages and perform actions with/on them. 
"""
import logging
import os

import sqlite3
import pandas as pd
from pyromancy.settings import (
    MESSAGES, CHANNELS, DBSUBSCRIBER_ACCUM_BATCH_SIZE, RUNTIME_MESSAGE_NAMES
)
import torch


class Subscriber(object):
    """
    A Subscriber represents the listeners that just do stuff with the compute
    stream and do not emit messages about it. For example, they could log messages
    to a file, to a database, or Tensorboard.  They could also save the model, etc.
    """
    def __init__(self):
        self.trial_path = None
        
    def receive(self, message, channel):
        """
        Args:
            message (dict): The dictionary of
                {'message_name': message_name,
                 'message_value': message_value}
                which was published by a Publisher
            channel (str): the name of the channel upon which the message was
                emitted.  At the moment, all subscribers hear all channels.
                The filter lies in the object having the proper method to
                listen to the message.
        """
        if hasattr(self, channel):
            channel_func = getattr(self, channel)
            channel_func(message)

    def metric_events(self, message):
        pass

    def actuator_events(self, message):
        pass

    def update(self, experiment):
        self.trial_path = experiment.trial_path

        if self.trial_path is None:
            raise Exception("Subscribers needs a valid trial path: ",
                            self.trial_path)


class Echo(Subscriber):
    """
    The most basic Subscriber.  It is mainly used for debugging.  For actual
    printed statements during the training, please use LogSubscriber.
    """

    def training_events(self, message):
        print(message)

    def metric_events(self, message):
        print(message)

    def actuator_events(self, message):
        print(message)


class DBSubscriber(Subscriber):
    """
    The DBSubscriber subscribes to metric events and logs them into the
    target database. It can also save the parameters that were used in each
    training instantiation.

    .. seealso:: For how pandas interacts with sqlite3, please visit
        https://www.dataquest.io/blog/python-pandas-databases/

    Attributes:
        _train_event_accum (list)
        _metric_event_accum (list)
        _db_path (str)
        _conn (sqlite.DBConnection)
    """
    def __init__(self):
        super(DBSubscriber, self).__init__()
        self._train_event_accum = []
        self._metric_event_accum = []

    def update(self, experiment):
        super(DBSubscriber, self).update(experiment)
        self._db_path = os.path.join(self.trial_path, "events.db")
        self._conn = sqlite3.connect(self._db_path)

    def _persist(self, data, table_name):
        pd.DataFrame(data).to_sql(
                name=table_name, 
                con=self._conn, 
                if_exists="append"
            )

    def flush_training_events(self):
        if len(self._train_event_accum) > 0:
            self._persist(self._train_event_accum, "training_events")
            self._train_event_accum = []

    def flush_metric_events(self):
        if len(self._metric_event_accum) > 0:
            self._persist(self._metric_event_accum, "metric_events")
            self._metric_event_accum = []
        
    def training_events(self, message):
        """Handle the training events

        Args:
            message (dict): Training events are dictionaries of the format
                {
                    'event_name': name_of_training_event, 
                    'event_value': value
                }
        """
        self._train_event_accum.append(message)

        if message['event_name'] in (MESSAGES.EPOCH_END, MESSAGES.EXPERIMENT_END):
            self.flush_training_events()
            self.flush_metric_events()

        if len(self._train_event_accum) > DBSUBSCRIBER_ACCUM_BATCH_SIZE:
            self.flush_training_events()

    def metric_events(self, message):
        """Handle metric events.  
        
        Metrics will be assumed to be grouped.

        Args:
            message (dict)
        """        
        self._metric_event_accum.append(message)

        if len(self._metric_event_accum) > DBSUBSCRIBER_ACCUM_BATCH_SIZE:
            self.flush_metric_events()

    def hyperparam_events(self, message):
        """
        """
        self._persist([message], "hyperparam_events")


class LogSubscriber(Subscriber):
    def __init__(self, to_file=True, to_console=True, log_level=logging.INFO):
        super(LogSubscriber, self).__init__()

        self._to_file = to_file
        self._to_console = to_console
        self.logger = logging.getLogger("pyromancy")
        self.logger.propagate = False
        self.logger.setLevel(log_level)


    def update(self, experiment):
        super(LogSubscriber, self).update(experiment)
        self.update_handlers()

    def update_handlers(self):
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        formatter = logging.Formatter(
            "[%(experiment_name)s %(trial_name)s][%(asctime)s]"
            "[%(event_type)s epoch=%(epoch)d split=%(split)s] %(message)s"
        )

        if self._to_file:
            file_handler = logging.FileHandler(os.path.join(self.trial_path, 
                                                            "events.log"))
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        if self._to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def write(self, message, event_type):
        message = dict(message.items())
        extra = {key: message.pop(key) for key in ['experiment_name', 
                                                   'trial_name', 
                                                   'epoch', 
                                                   'split']}
        extra['event_type'] = event_type

        message_flat = ", ".join([f"{key}::{value}" 
                                 for key, value in message.items()])
        
        self.logger.info(message_flat, extra=extra)

    def training_events(self, message):
        self.write(message, CHANNELS.TRAINING_EVENTS)

    def metric_events(self, message):
        self.write(message, CHANNELS.METRIC_EVENTS)

    def hyperparam_events(self, message):
        self.write(message, CHANNELS.HYPERPARAM_EVENTS)


class TrainState(Subscriber):
    def __init__(self, model=None, model_checkpoint_filename='model.pth', 
                 target_split='val', target_metric='loss', mode='min'):
        super(TrainState, self).__init__()

        self._metrics = {"train": {}, 
                         "val": {},
                         "test": {},
                         "val*": {}}

        self.model = model
        self.model_checkpoint_filename = model_checkpoint_filename
        self.target_split = target_split
        self.target_metric = target_metric
        self.mode = mode
        self.patience = 0

        if self.mode == 'min':
            self.best_value = 10**7
        elif self.mode == 'max':
            self.best_value = -10**7
        else:
            raise Exception(f"Unknown mode: {self.mode}")

    def _init_metric(self, split, metric_name):
        self._metrics[split][metric_name] = {
            "running": 0,
            "history": [],
            "count": 0
        }

    def _update_metric(self, split, metric_name, metric_value):
        if metric_name not in self._metrics[split]:
            self._init_metric(split, metric_name)
        metric = self._metrics[split][metric_name]
        metric['count'] += 1
        metric['running'] += (metric_value - metric['running']) / metric['count']

    def value_of(self, split, metric_name):
        return self._metrics[split][metric_name]['running']

    def __getitem__(self, key):
        split, metric_name = key.split(".")
        return self.value_of(split, metric_name)

    def save_model(self):
        if self.model is not None:
            full_path = os.path.join(self.trial_path, 
                                     self.model_checkpoint_filename)
            torch.save(self.model.state_dict(), full_path)

    def reload_best(self):
        if self.model is not None:
            full_path = os.path.join(self.trial_path, 
                                     self.model_checkpoint_filename)
            if os.path.exists(full_path):
                self.model.load_state_dict(torch.load(full_path))

    def training_events(self, message):
        if message['event_name'] == MESSAGES.EPOCH_END:
            checkpoint_value = self.value_of(self.target_split, self.target_metric)

            for metric_dicts in self._metrics.values():
                for metric in metric_dicts.values():
                    metric['history'].append(metric['running'])
                    metric['running'] = 0
                    metric['count'] = 0

            min_condition = (
                self.mode == 'min' and checkpoint_value < self.best_value
            )
            max_condition = (
                self.mode == 'max' and checkpoint_value > self.best_value
            )

            if min_condition or max_condition:
                self.best_value = checkpoint_value
                self.save_model()
                self.patience = 0
            else:
                self.patience += 1

    def metric_events(self, message):
        split = message['split']
        for metric_name, metric_value in message.items():
            if metric_name in RUNTIME_MESSAGE_NAMES:
                continue
            self._update_metric(split, metric_name, metric_value)
