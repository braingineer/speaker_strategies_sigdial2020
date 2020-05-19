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
    
    This implementation currently encodes two ideas:
        
        1. Rational Speech Acts (RSA) nested interlocutors
        2. Within RSA, an Out-Of-Context computation
        
    The algorithm leaves open how the model's output is computed,
    but given that output and two keys ('psi_value' and 'S0_probability'),
    a recursive hierarchy of speakers and listeners is computed:
        
        Speakers: S0, S1, S2 
        Listeners: L0, L1, L2
        
    The way the model's output is computed is the difference between context
    free and context sensitive as it has been construed here. Specifically,
    context free does not push the alternative objects into the semantic computations
    while context sensitive does. 
        
    Attributes:
        model (torch.nn.Module): one of the LUX Models. LUX 2A and 2C are currently
            not supported, so stick with 2B (sometimes just called Model B).
            See `magis.models.model_b`.
    """
    def __init__(self, model, detach_tensors=True, **kwargs):
        self.model = model
        self.detach_tensors = detach_tensors
        
    def validate(self):
        pass
        
    def _compute_model_output(self, context, psi_kwargs):
        raise NotImplementedError
     
    def generate_and_interpert(self, context, rsa_alpha=1.0):
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
        sem_probas = model_output['psi_value']
        
        l0_probas = nan_correction(normalize(sem_probas, dim=OBJECT_DIM))
        s0_probas = model_output['S0_probability']
        
        
        # l0 + s0 -> s1
        s1_probas_v1 = normalize(l0_probas**rsa_alpha * s0_probas, dim=LANG_DIM)
        s1_probas_v2 = normalize(l0_probas**rsa_alpha * sem_probas * avails, dim=LANG_DIM)
        # PRIORS WOULD GO HERE
        l2_probas = normalize(s1_probas_v1 * context.target_prior.unsqueeze(dim=2), dim=OBJECT_DIM)
        
        
        l1_probas = nan_correction(normalize(
            s0_probas * context.target_prior.unsqueeze(dim=2), 
            dim=OBJECT_DIM
        ))
        s2_probas_v1 = normalize(l1_probas**rsa_alpha * s0_probas, dim=LANG_DIM)
        s2_probas_v2 = normalize(l1_probas**rsa_alpha * sem_probas * avails, dim=LANG_DIM)
        
        # sem
        context.SEM_probabilities = sem_probas
        
        # level 0
        # use no ooc for l0 no matter what since we don't want to steal probability mass in interp
        context.L0_probabilities = l0_probas
        
        if use_utterance_indices:
            context.L0_probabilities_full = context.L0_probabilities
            context.L0_probabilities = context.L0_probabilities.gather(index=utterance_indices, dim=2)
            
        context.S0_probabilities = s0_probas
        
        # level 1
        if use_utterance_indices:
            context.L1_probabilities_full = l1_probas
            context.L1_probabilities = l1_probas.gather(index=utterance_indices, dim=2)
        else:
            context.L1_probabilities = l1_probas
            

        context.S1_probabilities = s1_probas_v1
        context.S1_probabilities_v2 = s1_probas_v2
            
        context.L2_probabilities = l2_probas
            
        if use_utterance_indices:
            context.L2_probabilities_full = context.L2_probabilities
            context.L2_probabilities = context.L2_probabilities.gather(index=utterance_indices, dim=2)
            
        context.S2_probabilities = s2_probas_v1
        context.S2_probabilities_v2 = s2_probas_v2
        
        return context
    
    def __call__(self, *args, **kwargs):
        """ Shortcut for `self.generate_and_interpet` """
        return self.generate_and_interpert(*args, **kwargs)

    
class ContextFreeAlgorithm(LanguageUseAlgorithms):
    def _compute_model_output(self, context):
        batch, num_obj, feature_size = context.object_features.shape
        model_output = self.model(
            context.object_features.view(batch * num_obj, feature_size)
        )
        if self.detach_tensors:
            model_output = {k:v.detach() for k,v in model_output.items()}
        
        return {
            'psi_value': model_output['psi_value'].view(batch, num_obj, -1),
            'S0_probability': model_output['S0_probability'].view(batch, num_obj, -1)
        }
      

class LockedAlpha(ContextFreeAlgorithm):
    def __init__(self, model, detach_tensors=True, rsa_alpha=1.0, **kwargs):
        super().__init__(model, detach_tensors)
        self.rsa_alpha = rsa_alpha
        
    def __call__(self, *args, **kwargs):
        kwargs['rsa_alpha'] = self.rsa_alpha
        return super().__call__(*args, **kwargs)
        