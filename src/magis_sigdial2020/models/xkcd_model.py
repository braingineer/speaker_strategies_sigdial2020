import argparse
import os
import yaml

from magis_sigdial2020.modules.encoders import MLPEncoder
from magis_sigdial2020.utils.model import reload_trial_model
from magis_sigdial2020.utils.nn import new_parameter
import torch
import torch.nn as nn
import torch.nn.functional as F


torch_safe_log = lambda x: torch.log(torch.clamp(x, 1e-20, 1e7))


class XKCDModel(nn.Module):
    MODEL_TYPE = 'semantic'
    
    @classmethod
    def from_pretrained(cls, dirpath):
        with open(os.path.join(dirpath, 'hparams.yaml')) as fp:
            hparams = argparse.Namespace(**yaml.load(fp, Loader=yaml.FullLoader))
        hparams.trial_path = dirpath
        return cls.make(hparams, reload=True, eval_mode=True)
    
    @classmethod
    def make(cls, hparams, reload=False, eval_mode=False):
        model = cls(
            input_size=hparams.input_size,
            encoder_size=hparams.encoder_size,
            encoder_depth=hparams.encoder_depth,
            prediction_size=hparams.prediction_size
        )
        if reload:
            reload_trial_model(model, hparams.trial_path)
        if eval_mode:
            model = model.eval()
        return model
    
    def __init__(self, input_size, encoder_size, encoder_depth, prediction_size):
        super(XKCDModel, self).__init__()
        self.encoder = MLPEncoder(size_in=input_size,
                                  layer_sizes=[encoder_size]*encoder_depth,
                                  add_final_nonlinearity=True)
        self.decoder = nn.Linear(in_features=encoder_size, out_features=prediction_size)
        self.availabilities = new_parameter(1, prediction_size)
    
    def forward(self, x_input):
        output = {}
        
        x_encoded = self.encoder(x_input)
        output['phi_logit'] = self.decoder(x_encoded)
        
        output['log_word_score'] = (
            torch_safe_log(torch.sigmoid(output['phi_logit'])) 
            + torch.log(torch.sigmoid(self.availabilities))
        )
        
        output['word_score'] = torch.exp(output['log_word_score'])
        output['S0_probability'] = F.softmax(output['log_word_score'], dim=1)

        return output
    
class XKCDModelWithRGC(XKCDModel):
    """
    XKCD Model with Referential Goal Composition (RGC). RGC is the algorithm for
    combining semantic computations to compute the probability of a target referent
    and none of the distractors.
    
    In the single object case, this model will behave exactly as the base XKCDModel.
    When alternate objects as passed into the model's call as additional arguments,
    the first is considered the target and the remainder are considered distractors.
    
    Example:
        
        modelb = XKCDModelWithRGC.from_pretrained(pretrained_dir_path)
        x0, x1, x2 = Context.from_cic_row(cic, 0)
    """
    def forward(self, x_target, *x_alts):
        x_target_encoded = self.encoder(x_target)
        x_alts_encoded = list(map(self.encoder, x_alts))
        
        phi_target_logit = self.decoder(x_target_encoded)
        phi_target = torch.sigmoid(phi_target_logit)
        
        if len(x_alts) > 0:
            # shape=(num_alts, batch_size, 829)
            phi_alt = torch.stack([
                torch.sigmoid(self.decoder(x_alt_i_encoded))
                for x_alt_i_encoded in x_alts_encoded
            ])
            phi_alt, _ = torch.max(phi_alt, dim=0)
            # This operation can be understood in multiple ways (tnorms, scales, cdf)
            # The interpretation I am choosing is set restriction on the
            # CDF: P(alt < T < target) = P(T < target) - P(T < alt)
            # The relu is there to guarantee positiveness (obviously if target < alt, the CDF is 0)
            psi_value = F.relu(phi_target - phi_alt)
        else:
            phi_alt = torch.zeros_like(phi_target)
            psi_value = phi_target
            
        word_score = psi_value * torch.sigmoid(self.availabilities)
        S0_probability = word_score / word_score.sum(dim=1, keepdim=True)
            
        return {
            'phi_logit': phi_target_logit,
            'phi_target': phi_target,
            'phi_alt': phi_alt,
            'psi_value': psi_value,
            'word_score': word_score,
            'S0_probability': S0_probability
        }
