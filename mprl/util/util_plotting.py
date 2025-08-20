import numpy as np
import matplotlib.pyplot as plt
import torch
from util_sampling_split import get_splits

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


def plot_index_histogram(split_strategy: dict={'correction_completion': 'current_idx', 'mean_std': [10, 5],
                                               'n_splits': 8, 'q_loss_strategy': 'continuing',
                                               'random_permute_splits': False,
                                               'ranges': [50, 35, 10, 5], 'size_exception_for_first_and_last_split': True,
                                               'size_range': [10, 20], 'split_strategy': 'random_size_range',
                                               'v_func_estimation': 'truncated'},
                         n_samples=100000):
    times = torch.linspace(0, 2, 100)[None, :]
    total_hits = np.zeros(101)  #1 extra for empty last splits

    for n in range(n_samples):
        hits = np.array(get_splits(times, split_strategy))
        start_hits = [0] * len(hits)
        for i in range(1, len(hits)):
            start_hits[i] = hits[i-1] + start_hits[i - 1]

        total_hits[start_hits] += 1

    # first split always starts at t=0 -> not interesting, also additional zero splits at end not interesting
    total_hits[-1] = 0
    total_hits[0] = 0
    #plt.xticks(np.arange(len(total_hits)))
    if split_strategy["split_strategy"] == "random_size_range":
        plt.title(split_strategy["split_strategy"] +f" range: {split_strategy['size_range']} n_split={split_strategy['n_splits']}")
    elif split_strategy["split_strategy"] == "fixed_size_rand_start":
        plt.title(split_strategy[
                      "split_strategy"] + f" fixed_size: {split_strategy['fixed_size']}")
    elif split_strategy["split_strategy"] == "rand_semi_fixed_size_focus_on_region":
        plt.title(f" fixed_size: {split_strategy['fixed_size']}, fr={split_strategy['focus_regions']},"
                                          f" fr_region_size={split_strategy['focus_regions_size_1way']}), "
                                          f"fr_sample_min={split_strategy['min_decreased_size']}, ")
    plt.bar(np.arange(len(total_hits)), total_hits)
    plt.show()

split_strategy={'correction_completion': 'current_idx', 'mean_std': [10, 5],
                                               'n_splits': 8, 'q_loss_strategy': 'continuing',
                                               'random_permute_splits': False,
                                               'ranges': [50, 35, 10, 5], 'size_exception_for_first_and_last_split': True,
                                               'size_range': [10, 20], 'split_strategy': 'random_size_range',
                                               'v_func_estimation': 'truncated'}
'''
for i in range(6,10):
    split_strategy["n_splits"] = i
    plot_index_histogram(split_strategy)
'''
split_strategy["split_strategy"] = "fixed_size_rand_start"
split_strategy["fixed_size"] = 20



#sizes = [5,10,15,20,25,30]
#for i in range(len(sizes)):
#    split_strategy["fixed_size"] = sizes[i]
#    plot_index_histogram(split_strategy)

split_strategy["fixed_size"] = 20
split_strategy["split_strategy"] = "rand_semi_fixed_size_focus_on_region"
split_strategy["focus_regions"]= [50]
split_strategy["focus_regions_size_1way"] = 25
split_strategy["min_decreased_size"] = 10
split_strategy["n_splits"] = 15
sizes = [10,15,20,25,30,35,40,45,50]
#for i in range(len(sizes)):
    #split_strategy["min_decreased_size"] = sizes[i]
    #split_strategy["focus_regions_size_1way"] = sizes[i]
plot_index_histogram(split_strategy)