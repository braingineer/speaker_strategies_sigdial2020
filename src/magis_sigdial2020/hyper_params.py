import argparse
import itertools
import json

import numpy as np
import yaml


def verify_key(key, dict_):
    assert key in dict_, f"Error: {key} required"
    
    
def verify_options(item, options):
    if not isinstance(options, set):
        options = set(options)
    assert item in options

    
def format_runtime_strings(input, runtime_dict):
    if isinstance(input, dict):
        output_dict = {}
        for key, value in list(input.items()):
            output_dict[key] = format_runtime_strings(value, runtime_dict)
        return output_dict
    elif isinstance(input, (list,tuple)):
        out = []
        for input_i in input:
            out.append(format_runtime_strings(input_i, runtime_dict))
        return type(input)(out)
    elif isinstance(input, str):
        return input.format(**runtime_dict)
    else:
        return input
    
    
class HyperParameters(argparse.Namespace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, 'hp_search'):
            self.search = HyperParameterSearch()
            self.search_hp = []
            for hp_spec in self.hp_search:
                verify_key("enabled", hp_spec)
                verify_options(hp_spec["enabled"], [False, True])
                if hp_spec.pop("enabled") is False:
                    continue
                verify_key("name", hp_spec)
                verify_key("search_type", hp_spec)
                self.search_hp.append(hp_spec["name"])
                if hp_spec["search_type"] == "static":
                    verify_key("values", hp_spec)
                    self.search.add_list(
                        name=hp_spec["name"], 
                        values=hp_spec["values"]
                    )
                else:
                    verify_options(hp_spec["search_type"], 
                                   ["linear", "exponential", "linearly_spaced"])
                    self.search.add(
                        name=hp_spec.pop("name"),
                        search_type=hp_spec.pop("search_type"),
                        **hp_spec
                    )
            if hasattr(self, 'hp_search_constraints'):
                for constraint in self.hp_search_constraints:
                    self.search.add_constraint(**constraint)
        
    @classmethod
    def load(cls, filepath, runtime_dict=None):
        with open(filepath) as fp:
            hparams_dict = yaml.load(fp, Loader=yaml.FullLoader)
        if runtime_dict is not None:
            hparams_dict = format_runtime_strings(hparams_dict, runtime_dict)
        hparams_dict["hparams_load_path"] = filepath
        return cls(**hparams_dict)
            
    
    def get_serializable_contents(self):
        contents = {}
        for key, value in vars(self).items():
            if key in set(["search", "constraints"]): 
                continue
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            
            assert isinstance(value, (float, int, str, bool))
            contents[key] = value
        return contents
    
    def save(self, filepath):
        contents = self.get_serializable_contents()
        with open(filepath, "w") as fp:
            yaml.dump(contents, fp)


class HyperParameterSearch:
    def __init__(self):
        self._hp = {}
        self.constraints = []

    def add(self, name, search_type, **search_kwargs):
        assert hasattr(self, f'_make_{search_type}'), f'Unknown search type: {search_type}'
        search_value_func = getattr(self, f'_make_{search_type}')
        values = search_value_func(**search_kwargs)
        self._add(name, search_type, values, **search_kwargs)
        return self

    def _add(self, name, search_type, values, **search_kwargs):
        self._hp[name] = {
            "name": name,
            "values": values,
            "search_type": search_type
        }
        self._hp[name].update(search_kwargs)

    def add_list(self, name, values):
        self._add(name, "static_list", values)
        return self

    def _make_linear(self, min_val, max_val, step_size):
        out = []
        while min_val <= max_val:
            out.append(min_val)
            min_val += step_size
        return out
    
    def _make_linearly_spaced(self, min_val, max_val, num_steps):
        return np.linspace(min_val, max_val, num_steps).tolist()
        
    def _make_exponential(self, min_val, max_val, step_size):
        out = []
        while min_val <= max_val:
            out.append(min_val)
            min_val *= step_size
        return out
    
    def add_constraint(self, **constraint):
        self.constraints.append(
            lambda hp_dict: all([hp_dict[k] == v for k, v in constraint.items()])
        )

    def make_sets(self):
        # expand out the hyper parameters
        productable_list = []
        for name, hp_dict in self._hp.items():
            productable_list.append([(name, value) for value in hp_dict['values']])
        # each item in hp_product is [(name0, value), (name1, value), ...]
        hp_sets = itertools.product(*tuple(productable_list))
        # mapping dict to each of those lists of tuples will create a list of dicts
        hp_sets = list(map(dict, hp_sets))

        for constraint in self.constraints:
            hp_sets = [hp_set for hp_set in hp_sets if not constraint(hp_set)]
        return hp_sets
    
    def expand_base_hparams(self, base_hparams):
        HParamsClass = base_hparams.__class__
        hp_sets = self.make_sets()
        all_expanded_hparams = []
        for hp_set in hp_sets:
            expanded_hparams = HParamsClass(**vars(base_hparams))
            for key, value in hp_set:
                setattr(expanded_hparams, key, value)
            all_expanded_hparams.append(expanded_hparams)
        return all_expanded_hparams