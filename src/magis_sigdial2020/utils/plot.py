from magis_sigdial2020.datasets.colorspace import get_colorspace
from magis_sigdial2020.datasets.xkcd import XKCD
from magis_sigdial2020.utils import color
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_one(color_vector, color_space='xy', ax=None, title=""):
    if color_space == 'xy':
        rgb_vector, _ = color.xy2rgbhsv(color_vector)
    elif color_space == 'hsl':
        rgb_vector = color.hsl2rgb(*color_vector)
    elif color_space == 'hsv':
        rgb_vector = color.hsv2rgb(*color_vector)
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.add_patch(plt.Rectangle((0, 0), 20, 20, color=tuple(rgb_vector)))
    ax.axis('off')
    ax.set_title(title)

    
def plot_three(color_vector1, color_vector2, color_vector3, title='', color_space='hsv', axes=None):
    if axes is None:
        _, axes = plt.subplots(1, 3)
    else:
        assert len(axes) == 3, "Need axes with 3 subplots"
    plot_one(color_vector1, color_space=color_space, ax=axes[0], title=f'target ({title})')
    plot_one(color_vector2, color_space=color_space, ax=axes[1], title=f'alt ({title})')
    plot_one(color_vector3, color_space=color_space, ax=axes[2], title=f'alt ({title})')
    

def plot_row(row, transcript_title=False, subplot_top=0.8):
    """Plot a row from the CIC data frame
    
    Assumption: 
        The color values in row are still in HSL space
    """
    fig, axes = plt.subplots(1, 3)
    for object_name, ax in zip(['target', 'alt1', 'alt2'], axes):
        plot_one(color_vector=row[object_name], color_space='hsl', ax=ax, title=object_name)
    success_string = "successful" if row.clicked == 'target' else f"failed, clicked {row.clicked}"
    if transcript_title:
        transcript = "\n".join([f"[{i}] {ev['role'][0].upper()}: {ev['text']}" for i, ev in enumerate(row.utterance_events)])
        plt.suptitle(f"TRANSCRIPT [{success_string}]\n----------------- \n{transcript}", ha='left', x=0.1)
    elif isinstance(row.lux_label, str):
        plt.suptitle(f"matches lux label: {row.lux_label}; \nFull={row.utterance_events[0]['text']}")
    else:
        plt.suptitle(f"[pattern={row.utterance_events_pattern}] First is: {row.utterance_events[0]['text']}")
    plt.tight_layout()
    plt.subplots_adjust(top=subplot_top)
    

class ColorspacePlotter:
    def __init__(self, model, coordinate_system="fft", cuda=True):
        self.xkcd = XKCD.from_settings(coordinate_system=coordinate_system)
        self.csd = get_colorspace(coordinate_system=coordinate_system)
        self.device = ("cuda" if cuda and torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.p_word, self.phi = self.apply_model_to_colorspace()
        
        
        self._label2rows = {
            i:[] for i in self.xkcd.color_vocab._idx_to_token.keys()
        }
        for row_index, label_index in self.xkcd._target_fast:
            self._label2rows[label_index].append(row_index)
            
        self._label2rows = {
            label_index: np.array(row_indices, dtype=np.int32) 
            for label_index, row_indices in self._label2rows.items()
        }

    def apply_model_to_colorspace(self, eps=None):
        p_word = []
        phi = []
        to_numpy = lambda tensor: tensor.cpu().detach().numpy()
        
        batch_generator = self.csd.generate_batches(
            batch_size=256, 
            shuffle=False, 
            drop_last=False,
            device=self.device
        )
        
        for batch_index, batch in enumerate(batch_generator):
            model_output = self.model(batch['x_colors'])
            p_word.append(to_numpy(model_output['S0_probability']))
            phi.append(to_numpy(torch.sigmoid(model_output['phi_logit'])))
            
        p_word = np.vstack(p_word)
        p_word = p_word.reshape((self.csd.num_h, self.csd.num_s, self.csd.num_v, p_word.shape[-1]))
        phi = np.vstack(phi)
        phi = phi.reshape((self.csd.num_h, self.csd.num_s, self.csd.num_v, phi.shape[-1]))

        return p_word, phi

    def contour_plot(self, color_term, target='p_word', levels=[0.1, 0.5], 
                     linestyles=['--', '-'], figsize=(15, 5), title_prefix='', 
                     dim_reduce_func=np.mean, num_to_scatter=-1, scatter_seed=0):
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        activations = getattr(self, target)
        if len(title_prefix) > 0:
            title_prefix = title_prefix.strip() + ' '

        index = self.xkcd.color_vocab.lookup_token(color_term)
        
        
        if num_to_scatter > 0:
            random_state = np.random.RandomState(seed=scatter_seed)
            xkcd_row_indices = self._label2rows[index]
            subset = random_state.choice(
                xkcd_row_indices, 
                size=min(num_to_scatter, len(xkcd_row_indices)), 
                replace=False
            )
            full_scatter_points = self.xkcd._original_data_matrix[subset]

        im = axes[0].contourf(self.csd.h, self.csd.s, dim_reduce_func(activations, axis=2)[:, :, index].T, alpha=0.3)
        plt.colorbar(im, ax=axes[0])
        axes[0].set_xlabel("Hue")
        axes[0].set_ylabel("Saturation")
        if num_to_scatter > 0:
            axes[0].scatter(full_scatter_points[:, 0], full_scatter_points[:, 1], marker='x', color='black', alpha=0.5)
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)

        im = axes[1].contourf(self.csd.h, self.csd.v, dim_reduce_func(activations, axis=1)[:, :, index].T, alpha=0.3)
        plt.colorbar(im, ax=axes[1])
        axes[1].set_xlabel("Hue")
        axes[1].set_ylabel("Value")
        if num_to_scatter > 0:
            axes[1].scatter(full_scatter_points[:, 0], full_scatter_points[:, 2], marker='x', color='black', alpha=0.5)
        axes[1].set_xlim(0, 1)
        axes[1].set_ylim(0, 1)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        if target == "phi":
            plt.suptitle(f'Applicability (phi) of {color_term} across colorspace')
        elif target == "p_word":
            plt.suptitle(f'Probability of {color_term} across colorspace')

    def plot_both_contours(self, color_term, **kwargs):
        self.contour_plot(color_term, 'phi', **kwargs)
        self.contour_plot(color_term, 'p_word', **kwargs)
