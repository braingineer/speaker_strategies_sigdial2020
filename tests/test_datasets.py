from magis_sigdial2020.datasets.xkcd import XKCD
from magis_sigdial2020.datasets.xkcd.vectorized import get_xkcd_vocab
from magis_sigdial2020.datasets.color_reference_2017.vectorized import make_or_load_cic

XKCD_N_DATA = {
    "train": 1523108,
    "val": 108545,
    "test": 544764
}

XKCD_FEATURE_SIZES = {
    "fft": 54,
    "hsv": -1,
    "xy": -1
}

CIC_SPLITDIFFICULTY_COUNTS = {
    ('dev', 0): 7412,
    ('dev', 1): 499,
    ('dev', 2): 4951,
    ('dev', 3): 653,
    ('dev', 4): 1210,
    ('dev', 5): 962,
    ('test', 0): 8601,
    ('test', 1): 637,
    ('test', 2): 4057,
    ('test', 3): 596,
    ('test', 4): 1054,
    ('test', 5): 725,
    ('train', 0): 7788,
    ('train', 1): 566,
    ('train', 2): 4556,
    ('train', 3): 531,
    ('train', 4): 1367,
    ('train', 5): 876
}


def test_xkcd_from_settings():
    coordinate_system = "fft"
    
    xkcd = XKCD.from_settings(coordinate_system=coordinate_system)
    assert xkcd[0]["x_color_value"].shape[0] == XKCD_FEATURE_SIZES[coordinate_system]
    
    for split, size in XKCD_N_DATA.items():
        xkcd.set_split(split)
        assert len(xkcd) == size
    
def test_cic_from_settinsg():
    cic = make_or_load_cic()
    for split in ["train", "dev", "test"]:
        for difficulty in [0]:
            cic.set_split(split)
            cic.set_difficulty_subset([difficulty])
            cic.refresh_indices()
            assert len(cic) == CIC_SPLITDIFFICULTY_COUNTS[split, difficulty]