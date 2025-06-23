import numpy as np



def get_splits(times, split_strategy: dict={"split_strategy": "n_equal_splits", "n_splits": 1}):
    if split_strategy['split_strategy'] == "n_equal_splits":
        n_splits = int(split_strategy["n_splits"])
        default_split = times.size(-1) // n_splits

        split_size_list = [default_split] * n_splits
        if times.size(-1) > n_splits * default_split:
            for i in range(times.size(-1) - n_splits * default_split):
                split_size_list[-(i+1)] += 1


    #fixed max size does not make too much sense

    #elif split_strategy['split_strategy'] == "fixed_max_size":
    #    default_split = int(split_strategy["split_size"])
    #    if int(split_strategy["n_splits"]) * default_split < times.size(-1):
    #        raise Exception(f"Invalid split size {default_split} for the set amount of splits {split_strategy['n_splits']}. "
    #                        f"All {times.size(-1)} timesteps must be coverable in {split_strategy['n_splits']} splits with {default_split} elements each.")
    #    n_splits = times.size(-1) // default_split
    #    split_size_list = []
    #    remainder = times.size(-1)
    #    while np.sum(split_size_list) < times.size(-1):
    #        split_size_list.append(default_split)
    #        remainder -= default_split
    #
    #    if times.size(-1) > n_splits * default_split:
    #        split_size_list.append(times.size(-1) - n_splits * default_split)
    #    elif default_split > times.size(-1):
    #        split_size_list = [times.size(-1)]

    elif split_strategy["split_strategy"] == "random_size_range":
        min_size, max_size = split_strategy["size_range"]
        total_size_covered = 0
        split_size_list = []
        while total_size_covered < times.size(-1) and len(split_size_list) < int(split_strategy["n_splits"]) -1: #-1 because +1 element will be added in loop
            next_split = np.random.randint(min_size, max_size)
            if total_size_covered + next_split < times.size(-1):
                split_size_list.append(next_split)
                total_size_covered += next_split
            else:
                split_size_list.append(times.size(-1) - total_size_covered)
                total_size_covered = times.size(-1)
                break  # doesnt do anything that is done anyway, just for visual representation


        #add remaining split(s) so that format is always the same (--> always n splits)
        if total_size_covered == times.size(-1):
            split_size_list = split_size_list + [0] * (int(split_strategy["n_splits"]) - len(split_size_list))
        else: #-> did not reahc end in n_splits-1 steps --> put rest in nth bucket
            split_size_list.append(times.size(-1) - total_size_covered)

    elif split_strategy["split_strategy"] == "random_gauss":
        mean, std = split_strategy["mean_std"]
        total_size_covered = 0
        split_size_list = []
        while total_size_covered < times.size(-1) and len(split_size_list) < int(split_strategy["n_splits"]) -1:
            next_split = max(1, int(np.around(np.random.normal(float(mean), float(std)))))
            if total_size_covered + next_split < times.size(-1):
                split_size_list.append(next_split)
                total_size_covered += next_split
            else:
                split_size_list.append(times.size(-1) - total_size_covered)
                total_size_covered = times.size(-1)
                break  # doesnt do anything that isnt dont anyway, just for visual representation

        # add remaining split(s) so that format is always the same (--> always n splits)
        if total_size_covered == times.size(-1):
            split_size_list = split_size_list + [0] * (int(split_strategy["n_splits"]) - len(split_size_list))
        else:  # -> did not reahc end in n_splits-1 steps --> put rest in nth bucket
            split_size_list.append(times.size(-1) - total_size_covered)

    elif split_strategy["split_strategy"] == "alternating_ranges":
        ranges = split_strategy["ranges"]
        if len(ranges) != int(split_strategy["n_splits"]):
            raise Exception(f"len(ranges): {len(ranges)} UNEQUAL number of splits: {split_strategy['n_splits']}")
        split_size_list = ranges
        np.random.shuffle(split_size_list)

    else:
        print("Splitting strategy unknown: {}".format(split_strategy["split_strategy"]) + " possible strategies are "
                                                                                     '''"fixed_max_size":       split in segments, maximally as big as the arg
                                                                                     "n_equal_splits":       split into n equal parts (+one for the rest if num_samples is not a multiple)
                                                                                     "random_size_range":    randomized segment sizes, uniformly distributed from x to y (different values each call)
                                                                                     "random_gauss":         randomized segment sizes, gaussian distributed around mean with std
                                                                                     "alternating_ranges":   segment sizes of fixed sizes, but which policy part gets which segment size is randomized
                                                                                     -> for every strategy if it does not fully cover the data and the next segment would "overcover" it a smaller segment is added in the end
                                                                                     args:
                                                                                         split_size: int                #required for split strategy "fixed_max_size"
                                                                                         n_splits: int                  #ALWAYS REQUIRED gives total number of splits [does not forbit 0 size splits] (direct parameter for me"n_equal_splits")
                                                                                         size_range: [int,int]          #required for split_strategy "random_size_range"
                                                                                         mean_std: [int,int]            #required for split_strategy "random_gauss"
                                                                                         ranges: [int]*len(n_splits]    #required for split_strategy "alternating_ranges"
                                                                                         ''')
        raise KeyError
    if split_strategy["random_permute_splits"]:
        np.random.shuffle(split_size_list)
    return split_size_list