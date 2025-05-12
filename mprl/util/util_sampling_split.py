import numpy as np



def get_splits(times, split_strategy: dict={"split_strategy": "n_equal_splits", "n_splits": 1}):
    if split_strategy['split_strategy'] == "n_equal_splits":
        n_splits = int(split_strategy["n_splits"])
        default_split = times.size(-1) // n_splits

        split_size_list = [default_split] * n_splits
        if times.size(-1) > n_splits * default_split:
            split_size_list.append(times.size(-1) - n_splits * default_split)

    elif split_strategy['split_strategy'] == "fixed_max_size":
        default_split = int(split_strategy["split_size"])
        n_splits = times.size(-1) // default_split

        split_size_list = [default_split] * n_splits
        if times.size(-1) > n_splits * default_split:
            split_size_list.append(times.size(-1) - n_splits * default_split)
        elif default_split > times.size(-1):
            split_size_list = [times.size(-1)]

    elif split_strategy["split_strategy"] == "random_size_range":
        min_size, max_size = split_strategy["size_range"]
        total_size_covered = 0
        split_size_list = []
        while total_size_covered < times.size(-1):
            next_split = np.random.randint(min_size, max_size)
            if total_size_covered + next_split < times.size(-1):
                split_size_list.append(next_split)
            else:
                split_size_list.append(times.size(-1) - total_size_covered)
                break  # doesnt do anything that isnt done anyway, just for visual representation
            total_size_covered += next_split

    elif split_strategy["split_strategy"] == "random_gauss":
        mean, std = split_strategy["mean_std"]
        total_size_covered = 0
        split_size_list = []
        while total_size_covered < times.size(-1):
            next_split = max(1, int(np.around(np.random.normal(float(mean), float(std)))))
            if total_size_covered + next_split < times.size(-1):
                split_size_list.append(next_split)
            else:
                split_size_list.append(times.size(-1) - total_size_covered)
                break  # doesnt do anything that isnt dont anyway, just for visual representation
            total_size_covered += next_split

    else:
        print("Splitting strategy unknown: {}".format(split_strategy["split_strategy"]) + " possible strategies are "
                                                                                     '''"fixed_max_size":       split in segments, maximally as big as the arg
                                                                                     "n_equal_splits":       split into n equal parts (+one for the rest if num_samples is not a multiple)
                                                                                     "random_size_range":    randomized segment sizes, uniformly distributed from x to y (different values each call)
                                                                                     "random_gauss":         randomized segment sizes, gaussian distributed around mean with std
                                                                                 
                                                                                     -> for every strategy if it does not fully cover the data and the next segment would "overcover" it a smaller segment is added in the end
                                                                                     args:
                                                                                         split_size: int         #required for split strategy "fixed_max_size"
                                                                                         n_splits: int           #required for split_strategy "n_equal_splits"
                                                                                         size_range: [int,int]   #required for split_strategy "random_size_range"
                                                                                         mean_std: [int,int]     #required for split_strategy "random_gauss"
                                                                                         ''')
        raise KeyError

    return split_size_list