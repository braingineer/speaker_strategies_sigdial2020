from .experiment import Experiment
from . import reader
from . import settings


__ACTIVE_EXPERIMENT = None

def initialize(experiment_name, root_output_path=None, subscribers=None, 
               trial_name=None):
    """
    Initialize the experiment object (or overwrite current experiment object)

    The experiment will write results to: 
        root_output_path/experiment_name/trial_name

    Args:
        experiment_name (str): the name to give the experiment.  trials are
            grouped by experiment.  
        root_output_path(str): [default=None] the base directory to write outputs 
            This path should be static across the trials, or even static across 
            experiments if doing related experiments. 
            If nothing is passed / `root_output_path` is None, then the output
            path from `settings.get_root_output_path()` is used.  
        subscribers (list): [default=None] a list of 
            pyromancy.pyromq.MessageSubscriber objects that are added to the
            experiment. Each subscriber will receive the compute messages. 
        trial_name (str): [default=None] Normally, this is not set by the user
            and the library will generate a unique string as the trial name. 
            However, if resuming an experiment, the trial name would need
            to be passed in through this interface. 
            TODO: re-evaluate the usefulness of exposing trial_name here. 

    """
    global __ACTIVE_EXPERIMENT

    if root_output_path is None:
        root_output_path = settings.get_root_output_path()
    
    __ACTIVE_EXPERIMENT = Experiment(experiment_name=experiment_name, 
                                     root_output_path=root_output_path, 
                                     subscribers=subscribers)
    __ACTIVE_EXPERIMENT.set_active_trial(trial_name)
    
    return __ACTIVE_EXPERIMENT

def get_experiment():
    """ get the active experiment """
    if __ACTIVE_EXPERIMENT is None:
        raise Exception("No active experiment")
    return __ACTIVE_EXPERIMENT