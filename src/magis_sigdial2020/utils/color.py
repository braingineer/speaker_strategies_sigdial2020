import colorsys
import numpy as np

def hex2hsv(hex_string):
    if hex_string[0] == "#":
        hex_string = hex_string[1:]
    assert len(hex_string) == 6
    r = int(hex_string[:2], base=16) * 1.
    g = int(hex_string[2:4], base=16) * 1. 
    b = int(hex_string[4:], base=16) * 1.
    return rgb2hsv(r, g, b)

def rgb2hsv(r, g, b):
    if r > 1:
        r /= 255.
    if g > 1:
        g /= 255.
    if b > 1:
        b /= 255.
        
    return colorsys.rgb_to_hsv(r, g, b)

def hsl2rgb(h, s, l):
    # for whatever reason, python swaps hsl -> hls; 
    return colorsys.hls_to_rgb(h, l, s)

def hsv2rgb(h, s, v):
    if h > 1:
        h = h / 360.
    if s > 1:
        s = s / 100.
    if v > 1:
        v = v / 100.
    return colorsys.hsv_to_rgb(h, s, v)

def hsl2hsv(h, s, l):
    # for whatever reason, python swaps hsl -> hls; 
    return colorsys.rgb_to_hsv(*colorsys.hls_to_rgb(h, l, s))

def hsl2hsv_matrix(hsl_matrix):
    hsv_matrix = [hsl2hsv(*row_i) for row_i in hsl_matrix]
    return np.stack(hsv_matrix)

def xy2rgbhsv(color_tensor):
    if len(color_tensor.shape) == 1:
        pre = tuple()
    elif len(color_tensor.shape) == 2:
        pre = (slice(None),)
    elif len(color_tensor.shape) == 3:
        # assuming batch, 3 referents, color
        pre = (slice(None), slice(None),)
    else:
        raise Exception("Cannot handle ndim > 3")

    recon = np.arctan2(color_tensor[pre+(0,)], color_tensor[pre+(1,)]) / (2 * np.pi)
    if not isinstance(recon, np.ndarray):
        recon = np.array(recon)
    recon[recon < 0] += 1
    hsv_colors = np.zeros_like(color_tensor[pre+(slice(1, None),)])
    hsv_colors[pre+(slice(1, None),)] = color_tensor[pre+(slice(2, None),)]
    hsv_colors[pre+(0,)] = recon
    full_shape = hsv_colors.shape
    flat_color = hsv_colors.reshape(-1, 3)
    rgb_colors = np.stack([np.array(hsv_to_rgb(c[0], c[1], c[2])) for c in flat_color])
    rgb_colors.reshape(full_shape)
    return rgb_colors.squeeze(), hsv_colors.squeeze()

def hsv2fft(data, fft_resolution=3):
    if len(data.shape) == 1:
        data = data[None, :]
    
    assert len(data.shape) == 2, "Please input a matrix"
    assert data.shape[1] == 3, f"HSV only has 3 features, found {data.shape[1]}"
    
    resolutions = [fft_resolution] * 3
    gx, gy, gz = np.meshgrid(*[np.arange(r) for r in resolutions])
    data[:, 1:] /= 2
    arg = (np.multiply.outer(data[:, 0], gx) +
           np.multiply.outer(data[:, 1], gy) +
           np.multiply.outer(data[:, 2], gz))

    repr_complex = (
        np.exp(-2j * np.pi * (arg % 1.0))
        .swapaxes(1, 2)
        .reshape((data.shape[0], -1))
    )
    return np.hstack([repr_complex.real, repr_complex.imag]).astype(np.float32)