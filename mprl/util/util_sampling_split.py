import numpy as np
from pandas.core.tools import times


def get_splits(times, split_strategy: dict={"split_strategy": "n_equal_splits", "n_splits": 1}):
    '''
        computes splits of indexes giving times-tensor according to strategy defined in split_strategy
        times: must be whole sequence -> size(-1) must be number of env interactions

        return: split_size_list: list of number of time steps for each split
            (e.g. [25, 58, 17] -> use params p1 for first 25, then p2 for 58 then p3 for 17 steps)
    '''
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

        #allow first split to be smaller than min size
        if split_strategy["size_exception_for_first_and_last_split"]:
            # lower bound to ensure we dont have splits > max-size
            lower_bound_for_validity = times.size(-1) - (
                    split_strategy["n_splits"] - 1 - len(split_size_list)) * max_size
            upper_bound_for_validity = times.size(-1) - (split_strategy["n_splits"] -1) * min_size + 1
            next_split = np.random.randint(max(1,lower_bound_for_validity), min(max_size, upper_bound_for_validity))
            split_size_list.append(next_split)
            total_size_covered += next_split


        while total_size_covered < times.size(-1) and len(split_size_list) < int(split_strategy["n_splits"]) -1: #-1 because +1 element will be added in loop
            if not split_strategy["size_exception_for_first_and_last_split"]:
                # upper bound is so that remaining splits can support at least remaining_splits * min_split_size  #-1 for current value getting added
                upper_bound_for_validity = times.size(-1) - (
                            split_strategy["n_splits"] - 1 - len(split_size_list)) * min_size

            else:
                #upper bound so that remaining splits -1 are supported with at least min_split_size and the last one with at least 1  #-2 for current and last value
                upper_bound_for_validity = (times.size(-1) -1) - (
                        split_strategy["n_splits"] - 2 - len(split_size_list)) * min_size - total_size_covered #-2 because indexes are [0,99] not [1,100]

            #lower bound to ensure we dont have splits > max-size
            #lower_bound_for_validity = times.size(-1) - (
            #        split_strategy["n_splits"] - 1 - len(split_size_list)) * max_size
            lower_bound_for_validity = times.size(-1) - max_size  * (
                    split_strategy["n_splits"] - 1 - len(split_size_list)) - total_size_covered
            if max(min_size,lower_bound_for_validity) == min(max_size, upper_bound_for_validity):
                next_split = min(max_size, upper_bound_for_validity)
            else:
                next_split = np.random.randint(max(min_size,lower_bound_for_validity), min(max_size, upper_bound_for_validity))

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
    elif split_strategy["split_strategy"] == "fixed_size_rand_start":
        #randint between 1 and fixed_size [upper limit not include in randint; low <= x < high]
        split_size_list = [np.random.randint(1, split_strategy["fixed_size"]+1)]
        total_size_covered = split_size_list[0]

        split_size_list = split_size_list + [split_strategy["fixed_size"]] * ((times.size(-1) -total_size_covered) // split_strategy["fixed_size"])
        total_size_covered = total_size_covered + split_strategy["fixed_size"] * ((times.size(-1) -total_size_covered) // split_strategy["fixed_size"])
        if times.size(-1) != total_size_covered:
            split_size_list.append(times.size(-1) - total_size_covered)
        if len(split_size_list) < split_strategy["n_splits"]:
            split_size_list = split_size_list + [0] * (split_strategy["n_splits"] - len(split_size_list))
        '''
            focus_regions: [30, 80] #each value gives a seperate region 
                                #[focus_region[i]-focus_regions_size_1way, focus_region[i]+focus_regions_size_1way], 
                                #around where the sampling size is decreased 
                                #-- uniformly sampled in [min_decreased_size, fixed_size] (include_last=True) 
        focus_regions_size_1way: 20 #(half-)size of focus region
        min_decreased_size: 10  #min sampling distance in focus region
        '''
    elif split_strategy["split_strategy"] == "rand_semi_fixed_size_focus_on_region":
        focus_regions_start = np.array(split_strategy["focus_regions"]) - split_strategy["focus_regions_size_1way"]
        focus_regions_end = np.array(split_strategy["focus_regions"]) + split_strategy["focus_regions_size_1way"]
        fr_min = split_strategy["min_decreased_size"]

        split_size_list = [np.random.randint(0, split_strategy["fixed_size"])]
        total_size_covered = split_size_list[0]
        fr_idx = 0
        fr_reached = False
        next_potential_pos = total_size_covered + split_strategy["fixed_size"]
        while total_size_covered < times.size(-1):
            #make sure last segment is not too large [ignores focus_regions]
            if times.size(-1) -total_size_covered < split_strategy["fixed_size"]:
                split_size_list.append(times.size(-1) - total_size_covered)
                total_size_covered = times.size(-1)
            #ensure focus regions are sampled accordingly
            elif next_potential_pos > focus_regions_start[fr_idx] and total_size_covered < focus_regions_end[fr_idx]:
                fr_reached = True
                next_split = np.random.randint(fr_min, split_strategy["fixed_size"])
                split_size_list = split_size_list + [next_split]
                total_size_covered += next_split
            else:
                if fr_reached:
                    if fr_idx != len(focus_regions_start) - 1:
                        fr_idx += 1
                    fr_reached = False
                #get one more iteration before adding a fixed_size in case the next focus_region is already reached
                # --> "pass" and only in the next call we add the default size
                else:
                    jitter = 0 #int(np.random.normal(0, 2))
                    if total_size_covered + jitter + split_strategy["fixed_size"] > times.size(-1):
                        jitter = 0
                    split_size_list = split_size_list + [split_strategy["fixed_size"]+ jitter]
                    total_size_covered += split_strategy["fixed_size"] + jitter


        if len(split_size_list) < split_strategy["n_splits"]:
            split_size_list = split_size_list + [0] * (split_strategy["n_splits"] - len(split_size_list))

        assert len(split_size_list) <= split_strategy["n_splits"], (f""
                f"'n_splits'={split_strategy['n_splits']} is not sufficient for the given split_strategy "
                f"'rand_semi_fixed_size_focus_on_region' with fixed_size={split_strategy['fixed_size']} and "
                f"min_decreased_size={split_strategy['min_decreased_size']}. It requires {len(split_size_list)} or more splits.")






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

    elif split_strategy["split_strategy"] == "fixed_sizes":
        ranges = split_strategy["ranges"]
        if len(ranges) != int(split_strategy["n_splits"]):
            raise Exception(f"len(ranges): {len(ranges)} UNEQUAL number of splits: {split_strategy['n_splits']}")
        if np.sum(ranges) != times.size(-1):
            raise Exception(f"Total sum of ranges {ranges} -> {np.sum(ranges)} does NOT fit (is unequal to) total timesteps {times.size(-1)}")
        split_size_list = ranges

    else:
        print("Splitting strategy unknown: {}".format(split_strategy["split_strategy"]) + " possible strategies are "
                                                                                     '''"fixed_max_size":       split in segments, maximally as big as the arg
                                                                                     "n_equal_splits":       split into n equal parts (+one for the rest if num_samples is not a multiple)
                                                                                     "random_size_range":    randomized segment sizes, uniformly distributed from x to y (different values each call)
                                                                                     "random_gauss":         randomized segment sizes, gaussian distributed around mean with std
                                                                                     "fixed_sizes":          segment sizes of fixed sizes
                                                                                     -> for every strategy if it does not fully cover the data and the next segment would "overcover" it a smaller segment is added in the end
                                                                                     args:
                                                                                         split_size: int                #required for split strategy "fixed_max_size"
                                                                                         n_splits: int                  #ALWAYS REQUIRED gives total number of splits [does not forbit 0 size splits] (direct parameter for me"n_equal_splits")
                                                                                         size_range: [int,int]          #required for split_strategy "random_size_range"
                                                                                         mean_std: [int,int]            #required for split_strategy "random_gauss"
                                                                                         ranges: [int]*len(n_splits]    #required for split_strategy "alternating_ranges"
                                                                                         ''')
        raise KeyError
    if split_strategy.get("random_permute_splits", False):
        np.random.shuffle(split_size_list)
    return split_size_list