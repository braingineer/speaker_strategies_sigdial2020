from collections import deque

import torch

def nan_correction(tensor):
    """ fix nan values by casting them to 0
    
    Note:
        this happens when 2 patches have nearly equally small values, 
        other patch has smaller value. then, all psi_values are 0.
        This affects the no ooc case when doing S1 from L0 and 
        all cases when doing L1 from S0. 
    """
    return torch.where(
        torch.isnan(tensor), 
        torch.zeros_like(tensor), 
        tensor
    )


def normalize(tensor, dim):
    return tensor / tensor.sum(dim=dim, keepdim=True)


class LanguageUseAlgorithms:
    """Using language to refer and understand reference
        
    Attributes:
        model (torch.nn.Module): one of the LUX Models. LUX 2A and 2C are currently
            not supported, so stick with 2B (sometimes just called Model B).
            See `magis.models.model_b`.
    """
    def __init__(self, model, detach_tensors=True, **kwargs):
        self.model = model
        self.detach_tensors = detach_tensors
        
    def validate(self):
        assert hasattr(self.model, 'MODEL_TYPE'), 'Use a model that specifies MODEL_TYPE'
        assert self.model.MODEL_TYPE == 'semantic', 'This implementation is for semantic models'
        
    def _compute_model_output(self, context, psi_kwargs):
        raise NotImplementedError
     
    def generate_and_interpert(self, context):
        """Compute the literal and pragmatic listeners and speakers. 
        
        Args:
            context (magis.utils.data.Context): A context object encodes the 
                communicative task: refer to a target object or interpret an utterance
                to resolve the target object. 
        """
        OBJECT_DIM = 1
        LANG_DIM = 2
        
        # model outputs
        model_output = self._compute_model_output(context)
        avails = torch.sigmoid(self.model.availabilities).view(1, 1, 829)
        if self.detach_tensors:
            avails = avails.detach()
        
        # setting up the utterance indices if they exist:
        use_utterance_indices = context.utterance_index.min().item() != -1
        if use_utterance_indices: 
            utterance_indices = context.utterance_index.view(-1, 1, 1).repeat((1, 3, 1))
            
        # semantic values
        context.SEM_probabilities = model_output['psi_value']
        
        # PRIOR WOULD GO HERE
        context.L0_probabilities = nan_correction(normalize(
            context.SEM_probabilities * context.target_prior.unsqueeze(dim=2), 
            dim=OBJECT_DIM
        ))
        if use_utterance_indices:
            context.L0_probabilities_full = context.L0_probabilities
            context.L0_probabilities = context.L0_probabilities.gather(
                index=utterance_indices, 
                dim=LANG_DIM
            )
        
        context.S0_probabilities = normalize(
            context.SEM_probabilities * avails, dim=LANG_DIM
        )
        
        return context
    
    def __call__(self, *args, **kwargs):
        """ Shortcut for `self.generate_and_interpet` """
        return self.generate_and_interpert(*args, **kwargs)


class NoContextAlgorithm(LanguageUseAlgorithms):
    def _compute_model_output(self, context):
        batch, num_obj, feature_size = context.object_features.shape
        model_output = self.model(
            context.object_features.view(batch * num_obj, feature_size),
        )
        
        if 'psi_value' not in model_output:
            raise RuntimeException("Please use with an RGC model that outputs `psi_value`")
        if "S0_probability" not in model_output:
            raise RuntimeException("Please use with an RGC model that outputs `S0_probability`")
            
        if self.detach_tensors:
            model_output = {k:v.detach() for k,v in model_output.items()}
        
        return {
            'psi_value': model_output['psi_value'].view(batch, num_obj, -1),
            'S0_probability': model_output['S0_probability'].view(batch, num_obj, -1)
        }


class RGCAlgorithm(LanguageUseAlgorithms):
    def _compute_model_output(self, context):
        batch, num_obj, feature_size = context.object_features.shape
        object_indices = deque(range(num_obj))
        output = {'psi_value': [], 'S0_probability': []}
        for _ in range(num_obj):
            model_output = self.model(
                *tuple(context.object_features[:, obj_i] for obj_i in object_indices),
            )
            if 'psi_value' not in model_output:
                raise RuntimeException("Please use with an RGC model that outputs `psi_value`")
            if "S0_probability" not in model_output:
                raise RuntimeException("Please use with an RGC model that outputs `S0_probability`")
            if self.detach_tensors:
                model_output = {k:v.detach() for k,v in model_output.items()}
            output['psi_value'].append(model_output['psi_value'])
            output['S0_probability'].append(model_output['S0_probability'])
            object_indices.rotate(-1)
            
        output = {k: torch.stack(v, dim=1) for k,v in output.items()}
        return output
