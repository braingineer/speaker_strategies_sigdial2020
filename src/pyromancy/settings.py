__ROOT_OUTPUT_PATH = "./pyromancy_output"
DBSUBSCRIBER_ACCUM_BATCH_SIZE = 50

class CHANNELS:
    """
    names for the channels the pyromancy will send compute messages on
    """
    TRAINING_EVENTS = 'training_events'
    METRIC_EVENTS = 'metric_events'
    HYPERPARAM_EVENTS = 'hyperparam_events'


class MESSAGES:
    """ names for the kinds of messages pyromancy will send """

    EXPERIMENT_START = "experiment.start"
    EXPERIMENT_END = "experiment.end"
    
    EPOCH_START = "epoch.start"
    EPOCH_END = "epoch.end"

    BATCH_START = "batch.start"
    BATCH_END = "batch.end"
    
    SET_SPLIT = "set.split"

RUNTIME_MESSAGE_NAMES = set(['epoch', 'split', 'timestamp', 
                             'experiment_name', 'trial_name'])

def set_root_output_path(root_output_path):
    global __ROOT_OUTPUT_PATH
    __ROOT_OUTPUT_PATH = root_output_path


def get_root_output_path():
    return __ROOT_OUTPUT_PATH
