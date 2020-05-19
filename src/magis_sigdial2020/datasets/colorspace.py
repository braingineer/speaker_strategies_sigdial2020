import numpy as np
from magis_sigdial2020.utils.data import Dataset

CSD = None
_cached_kwargs = None

def get_colorspace(**kwargs):
    global CSD, _cached_kwargs
    if CSD is None and _cached_kwargs is None:
        CSD = ColorSpaceDataset(**kwargs)
        _cached_kwargs = kwargs
    else:
        for k,v in kwargs.items():
            if _cached_kwargs[k] != kwargs[k]:
                CSD = ColorSpaceDataset(**kwargs)
                _cached_kwargs = kwargs
                break
    return CSD


class ColorSpaceDataset(Dataset):
    def __init__(self, num_h=40, num_s=20, num_v=20, coordinate_system='x-y'):
        self.num_h = num_h
        self.h = np.linspace(0, 1, num_h)
        self.num_s = num_s
        self.s = np.linspace(0, 1, num_s)
        self.num_v = num_v
        self.v = np.linspace(0, 1, num_v)
        
        self.colorspace = []
        hgrid, sgrid, vgrid = np.meshgrid(self.h, self.s, self.v,  indexing='ij')
        for h_i in range(num_h):
            for s_i in range(num_s):
                for v_i in range(num_v):
                    self.colorspace.append([hgrid[h_i, s_i, v_i], 
                                            sgrid[h_i, s_i, v_i], 
                                            vgrid[h_i, s_i, v_i]])
                    
        self.colorspace = np.array(self.colorspace, dtype=np.float32)
        
                
        self._colorspace = self.colorspace

        # xy 
        num_rows, _ = self._colorspace.shape
        hue_col = self._colorspace[:, 0]
        self.xy_colorspace = np.zeros((num_rows, 4), dtype=np.float32)
        self.xy_colorspace[:, 0] = np.sin(hue_col * 2 * np.pi)
        self.xy_colorspace[:, 1] = np.cos(hue_col * 2 * np.pi)    
        self.xy_colorspace[:, 2:] = self._colorspace[:, 1:]

        # fft
        colorspace = self._colorspace.copy()
        resolutions = [3, 3, 3] # usually controlled by fft_resolution.. see xkcd/vectorized
        gx, gy, gz = np.meshgrid(*[np.arange(r) for r in resolutions])
        colorspace[:, 1:] /= 2
        arg = (np.multiply.outer(colorspace[:, 0], gx) +
               np.multiply.outer(colorspace[:, 1], gy) +
               np.multiply.outer(colorspace[:, 2], gz))

        repr_complex = (
            np.exp(-2j * np.pi * (arg % 1.0))
            .swapaxes(1, 2)
            .reshape((colorspace.shape[0], -1))
        )
        self.fft_colorspace = np.hstack([repr_complex.real, repr_complex.imag]).astype(np.float32)
        
        if coordinate_system == 'x-y':
            self.colorspace = self.xy_colorspace
        elif coordinate_system == 'fft':
            self.colorspace = self.fft_colorspace
        
    def __getitem__(self, index):
        return {'x_colors': self.colorspace[index]}
    
    def __len__(self):
        return self.colorspace.shape[0]