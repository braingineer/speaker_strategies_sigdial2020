import copy

from magis_sigdial2020.utils.data import generate_batches
from magis_sigdial2020.utils.nn import RAdam
import numpy as np
import pyromancy
import pyromancy.subscribers as sub
import torch
import torch.optim as optim
import tqdm

    
class BaseTrainer:
    def __init__(self, hparams, bar_func=tqdm.tqdm, skip_initializing=True):
        self.hparams = hparams
        self._base_hparams = copy.deepcopy(hparams)
        
        self.bar_func = bar_func
        self.dataset = None
        self.model = None
        self.exp = None
        
        pyromancy.settings.set_root_output_path(self.hparams.root_output_path)
        
        if not skip_initializing:
            self.reset()

    def reset(self):
        # set seed in all possible places
        torch.manual_seed(self.hparams.seed)
        np.random.seed(self.hparams.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.dataset = self.make_dataset()
        self.model = self.make_model()
        self.train_state = self.make_train_state()
        self.exp = self.make_experiment()
        self.setup_loss()

    def make_dataset(self):
        raise NotImplementedError

    def make_model(self):
        raise NotImplementedError

    def setup_loss(self):
        pass

    def compute_model_output(self, batch_dict):
        raise NotImplementedError

    def compute_loss(self, batch_dict, model_output):
        raise NotImplementedError

    def compute_metrics(self, batch_dict, model_output):
        raise NotImplementedError
        
    def get_epochbar_stats(self, train_state):
        return {
            'val_loss': f'{train_state["val.loss"]:0.3f}'
        }
    
    def make_train_state(self):
        return sub.TrainState(
            model=self.model, 
            target_metric=self.hparams.checkpoint_metric
        )
    
    def make_experiment(self):
        return pyromancy.initialize(
            experiment_name=self.hparams.experiment_name,
            subscribers=[
                sub.DBSubscriber(),
                sub.LogSubscriber(to_console=False),
                self.train_state
            ],
            trial_name=getattr(self.hparams, "trial_name", None)
        )
    
    def log_hyperparameters(self):
        self.exp.log_hyperparams(self.hparams.get_serializable_contents())
        self.hparams.save(self.exp.expand_to_trial_path("hparams.yaml"))

    def run_hp_search_from_sets(self, hp_sets, verbose=False):
        for hp_set in hp_sets:
            self.hparams = copy.deepcopy(self._base_hparams)
            for name, value in hp_set.items():
                setattr(self.hparams, name, value)
                
            self.reset()
            if verbose:
                print(self.hparams)
                print(self.model)
                
            self._run(reset=False)
            
    def run_hp_search(self, verbose=False):
        assert hasattr(self.hparams, 'search'), "HyperParameter does not have hp_search"
        hp_sets = self.hparams.search.make_sets()
        if verbose:
            print(f"{len(hp_sets)} hyper parameter combinations")
        self.run_hp_search_from_sets(hp_sets, verbose=verbose)

    def run_once(self):
        self._run()

    def _run(self, reset=True):
        if reset:
            self.reset()

        hparams = self.hparams
        dataset = self.dataset
        model = self.model
        exp = self.exp
        train_state = self.train_state

        model = model.to(hparams.device)

        exp.log_exp_start()
        self.log_hyperparameters()

        optimizer = RAdam(model.parameters(),
                          lr=hparams.learning_rate,
                          weight_decay=getattr(hparams, 'weight_decay', 0.0))
        
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            verbose=True, 
            factor=getattr(hparams, 'schedule_anneal', 0.25), 
            patience=getattr(hparams, 'lr_scheduler_patience', hparams.patience_threshold//2)
        )
        epoch_bar = self.bar_func(total=hparams.num_epochs, desc=f'{hparams.experiment_name}/epochs', leave=False)

        dataset.set_split('train')
        train_bar = self.bar_func(total=len(dataset)//hparams.batch_size,
                                  desc='train', leave=True, ncols=0)

        dataset.set_split('val')
        val_bar = self.bar_func(total=len(dataset)//hparams.batch_size,
                                desc='val', leave=True, ncols=0)

        for _ in range(hparams.num_epochs):
            exp.log_epoch_start()

            model.train()
            dataset.set_split('train')
            exp.set_split('train')

            batch_generator = dataset.generate_batches(
                hparams.batch_size,
                device=hparams.device,
                num_workers=getattr(hparams, 'num_workers', 0)
            )
            train_bar.reset()
            for batch_index, batch_dict in enumerate(batch_generator):
                # gradient descent steps
                optimizer.zero_grad()
                model_output = self.compute_model_output(batch_dict)
                loss = self.compute_loss(batch_dict, model_output)
                loss.backward()
                optimizer.step()
                
                # metric tracking steps
                metrics = self.compute_metrics(batch_dict, model_output)
                metrics['loss'] = loss.item()
                exp.log_metrics(**metrics)

                # monitoring steps
                postfix_dict = {}
                for metric_name in metrics.keys():
                    postfix_dict[metric_name] = train_state[f"train.{metric_name}"]
                train_bar.set_postfix(**postfix_dict)
                train_bar.update()

            model.eval()
            dataset.set_split('val')
            exp.set_split('val')

            batch_generator = dataset.generate_batches(
                hparams.batch_size,
                device=hparams.device,
                num_workers=getattr(hparams, 'num_workers', 0)
            )
            val_bar.reset()
            for batch_index, batch_dict in enumerate(batch_generator):
                model_output = self.compute_model_output(batch_dict)
                loss = self.compute_loss(batch_dict, model_output)
                metrics = self.compute_metrics(batch_dict, model_output)
                metrics['loss'] = loss.item()
                exp.log_metrics(**metrics)

                postfix_dict = {}
                for metric_name in metrics.keys():
                    postfix_dict[metric_name] = train_state[f"val.{metric_name}"]
                val_bar.set_postfix(**postfix_dict)
                val_bar.update()
            
            val_loss = train_state["val.loss"]
            lr_scheduler.step(val_loss)
            
            epoch_bar_stats = {
                'lr': optimizer.param_groups[0]['lr']
            }
            epoch_bar_stats.update(self.get_epochbar_stats(train_state))
            exp.log_epoch_end()
            epoch_bar.set_postfix(patience=train_state.patience, **epoch_bar_stats)
            epoch_bar.update()
            if train_state.patience >= hparams.patience_threshold:
                break

        train_state.reload_best()
        model.eval()
        dataset.set_split('val')
        exp.set_split('val*')
        
        # not doing a shuffle=False, drop_last=False because XKCD is sorted by name
        # so some of the batches end up being really really bad
        # TODO: implement dataset-wide metrics by aggregating during batch iteration
        batch_generator = dataset.generate_batches(
            hparams.batch_size,
            device=hparams.device,
            num_workers=getattr(hparams, 'num_workers', 0)
        )
        val_bar.n = 0
        val_bar.refresh()

        for batch_index, batch_dict in enumerate(batch_generator):
            model_output = self.compute_model_output(batch_dict)
            loss = self.compute_loss(batch_dict, model_output)
            metrics = self.compute_metrics(batch_dict, model_output)
            metrics['loss'] = loss.item()
            exp.log_metrics(**metrics)

            postfix_dict = {}
            for metric_name in metrics.keys():
                postfix_dict[metric_name] = train_state[f"val*.{metric_name}"]
            val_bar.set_postfix(**postfix_dict)
            val_bar.update()

        exp.log_exp_end()
