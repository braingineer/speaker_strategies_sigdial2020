"""
Example:
    from pyromancy import Experiment, subscribers as sub

    exp = Experiment(experiment_name="example_experiment", 
                     root_output_path="./pyromancy_output", 
                     subscribers=[sub.DBSubscriber(), 
                                  sub.LogSubscriber()])
    exp.log_exp_start()
    for _ in range(num_epochs):
        exp.log_epoch_start()

        # ...

        for batch in batches:
            # ... 
            loss = measure_loss()
            exp.log_metric(loss=loss)
"""
import getpass
import os
import time

from pyromancy.settings import MESSAGES, CHANNELS
import pyromancy.utils as utils




class Experiment:
    def __init__(self, experiment_name, root_output_path, subscribers=None):
        if subscribers is None:
            subscribers = []
        self.subscribers = subscribers

        self.experiment_name = experiment_name
        self.root_output_path = root_output_path
        self._base_exp_path = os.path.join(self.root_output_path, 
                                           self.experiment_name)
        self.trial_path = None
        self._epoch = -1
        self._split = "unset"
        self.trial_name = "unset"

    def publish(self, channel, message):
        """Publish an event message to the subscribers

        Args:
            channel (str): the channel name which is supposed to receive the
                message.  Downstream, it is currently being used by subscribers
                as the name of a function.

                The following is True: `assert hasattr(subscriber, channel)`

                Example: if `channel = 'training_events'`, then subscribers who
                    would like to subscribe to `training_events` should have a function
                    named `training_events`

            message (dict): a dictionary which has following keys:
                {'message_name': 'some_name',
                 'message_value': 'some_value'}
        """
        message['epoch'] = self._epoch
        message['split'] = self._split
        message['timestamp'] = time.time()
        message['experiment_name'] = self.experiment_name
        message['trial_name'] = self.trial_name

        for subscriber in self.subscribers:
            subscriber.receive(message, channel)

    def add_subscriber(self, subscriber):
        """
        Add a subscriber to listen to the messages routed through this broker.

        Args:
            subscriber (pyromq.Subscriber): the subscriber to add
        """
        subscriber.update(self)
        self.subscribers.append(subscriber)

    def update_subscribers(self):
        for subscriber in self.subscribers:
            subscriber.update(experiment=self)

    def _new_trial(self):
        return f"{getpass.getuser()}_{os.getpid()}_{int(time.time())}"

    def set_active_trial(self, trial_name=None):
        if trial_name is None:
            trial_name = self._new_trial()
        self.trial_name = trial_name
        self.trial_path = os.path.join(self._base_exp_path, trial_name)
        utils.maybe_makedir(self.trial_path)
        self.update_subscribers()
        
    def expand_to_trial_path(self, filename):
        if self.trial_path is None:
            raise Exception("Initialize the experiment first")
        return os.path.join(self.trial_path, filename)

    def log_exp_start(self):
        self.publish(channel=CHANNELS.TRAINING_EVENTS, 
                     message={"event_name": MESSAGES.EXPERIMENT_START, 
                              "event_value": True})

    def log_exp_end(self):
        self.publish(channel=CHANNELS.TRAINING_EVENTS, 
                     message={"event_name": MESSAGES.EXPERIMENT_END, 
                              "event_value": True})

    def log_epoch_start(self):
        self._epoch += 1
        self.publish(channel=CHANNELS.TRAINING_EVENTS, 
                     message={"event_name": MESSAGES.EPOCH_START, 
                              "event_value": True})

    def log_epoch_end(self):
        self.publish(channel=CHANNELS.TRAINING_EVENTS, 
                     message={"event_name": MESSAGES.EPOCH_END, 
                              "event_value": True})

    def log_data_start(self):
        self.publish(channel=CHANNELS.TRAINING_EVENTS, 
                     message={"event_name": MESSAGES.DATA_START, 
                              "event_value": True})

    def log_data_end(self):
        self.publish(channel=CHANNELS.TRAINING_EVENTS, 
                     message={"event_name": MESSAGES.DATA_END, 
                              "event_value": True})

    def log_batch_start(self):
        self.publish(channel=CHANNELS.TRAINING_EVENTS, 
                     message={"event_name": MESSAGES.BATCH_START, 
                              "event_value": True})

    def log_batch_end(self):
        self.publish(channel=CHANNELS.TRAINING_EVENTS, 
                     message={"event_name": MESSAGES.BATCH_END, 
                              "event_value": True})

    def set_split(self, split):
        self._split = split
        self.publish(channel=CHANNELS.TRAINING_EVENTS, 
                     message={"event_name": MESSAGES.SET_SPLIT, 
                              "event_value": split})

    def log_metrics(self, **metrics):
        self.publish(channel=CHANNELS.METRIC_EVENTS, 
                     message=metrics)

    def log_hyperparams(self, hyperparams):
        self.publish(channel=CHANNELS.HYPERPARAM_EVENTS, 
                     message=hyperparams)