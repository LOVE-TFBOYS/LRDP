import random


def _with_optional(pair, fixed_seg=None, moving_seg=None, fixed_mask=None, moving_mask=None):
    if fixed_seg is not None:
        pair["fixed_seg"] = fixed_seg
    if moving_seg is not None:
        pair["moving_seg"] = moving_seg
    if fixed_mask is not None:
        pair["fixed_mask"] = fixed_mask
    if moving_mask is not None:
        pair["moving_mask"] = moving_mask
    return pair


def build_atlas_to_subject_pairs(image_list, atlas_path, seg_list=None, mask_list=None):
    pairs = []
    seg_map = dict(seg_list or [])
    mask_map = dict(mask_list or [])
    for image in image_list:
        if str(image) == str(atlas_path):
            continue
        pair = {"fixed": atlas_path, "moving": image}
        pairs.append(_with_optional(pair, seg_map.get(str(atlas_path)), seg_map.get(str(image)), mask_map.get(str(atlas_path)), mask_map.get(str(image))))
    return pairs


def build_all_pairs(image_list, exclude_self=True):
    pairs = []
    for fixed in image_list:
        for moving in image_list:
            if exclude_self and str(fixed) == str(moving):
                continue
            pairs.append({"fixed": fixed, "moving": moving})
    return pairs


def build_random_pairs(image_list, num_pairs, seed=42):
    all_pairs = build_all_pairs(image_list, exclude_self=True)
    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    return all_pairs[:num_pairs]


def build_subject_to_subject_pairs(image_list, mode="random", num_pairs=None, seed=42):
    if mode == "all":
        return build_all_pairs(image_list, exclude_self=True)
    if mode != "random":
        raise ValueError("mode must be 'random' or 'all'")
    if num_pairs is None:
        num_pairs = max(len(image_list) - 1, 1)
    return build_random_pairs(image_list, num_pairs, seed=seed)


def split_pairs(pairs, split_ratio=(0.7, 0.1, 0.2), seed=42):
    rng = random.Random(seed)
    pairs = list(pairs)
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])
    return {
        "train": pairs[:n_train],
        "val": pairs[n_train : n_train + n_val],
        "test": pairs[n_train + n_val :],
    }
