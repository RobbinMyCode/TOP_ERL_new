import numpy as np
import matplotlib.pyplot as plt


def plot_actions(times, actions, dims=[-1, -4], sample_idxes=[123,267,12], n_lines_per_plot=8):
    times = times[:,0].detach().cpu().numpy()
    actions = actions[:, 0, :, :].detach().cpu().numpy()

    if times.shape[:2] == actions.shape[:2]:
        times = times[0,:]

    n = 0
    while n < len(dims) * len(sample_idxes):
        for sample_idx in sample_idxes:
            for dim in dims:
                plt.plot(times, actions[sample_idx, :, dim], label=f"sample {sample_idx} dim={dim}")
                n += 1
                if n % n_lines_per_plot == 0 or n >= len(dims) * len(sample_idxes) -1:
                    plt.legend()
                    plt.show()

