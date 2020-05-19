from magis_sigdial2020.models.xkcd_model import XKCDModel, XKCDModelWithRGC
import torch

def test_xkcd_model():
    model = XKCDModel(input_size=54, encoder_size=4, encoder_depth=3, prediction_size=829)
    model_output = model(torch.randn((1,54)))
    assert model_output["phi_logit"].shape == (1, 829)
    assert model_output["log_word_score"].shape == (1, 829)
    assert model_output["word_score"].shape == (1, 829)
    assert model_output["S0_probability"].shape == (1, 829)
    
def test_xkcd_model_with_rgc():
    model = XKCDModelWithRGC(input_size=54, encoder_size=4, encoder_depth=3, prediction_size=829)
    model_output = model(torch.randn((1,54)))
    assert model_output["phi_logit"].shape == (1, 829)
    assert model_output["phi_target"].shape == (1, 829)
    assert model_output["phi_alt"].shape == (1, 829)
    assert model_output["psi_value"].shape == (1, 829)
    assert model_output["word_score"].shape == (1, 829)
    assert model_output["S0_probability"].shape == (1, 829)
    
    for i in [2, 3, 4]:
        model_inputs = tuple([torch.randn((1, 54)) for _ in range(i)])
        model_output = model(*model_inputs)
        assert model_output["phi_logit"].shape == (1, 829)
        assert model_output["phi_target"].shape == (1, 829)
        assert model_output["phi_alt"].shape == (1, 829)
        assert model_output["psi_value"].shape == (1, 829)
        assert model_output["word_score"].shape == (1, 829)
        assert model_output["S0_probability"].shape == (1, 829)
