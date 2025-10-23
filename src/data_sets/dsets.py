import copy
import functools
import logging
import os
import random

import torch
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from typing import List
import nibabel as nib
import numpy as np
from scipy import ndimage


from util.disk import getCache

log = logging.getLogger(__name__)

raw_cache = getCache("auto-qc")


@dataclass(order=True)
class CandidateInfoTuple:
    """Class for keeping track subject/session info."""

    qu_motion_float: float
    file_path: str
    subject_str: str
    session_str: str
    run_int: int = None
    suffix_str: str = None
    augmentation_index: int = None
    sort_index: float = field(init=False, repr=False)

    def __hash__(self):
        return hash(self.file_path)

    @property
    def subject(self) -> str:
        return self.subject_str

    def __post_init__(self):
        # sort by qu_motion_float
        self.sort_index = self.qu_motion_float

    @property
    def path_to_file(self) -> str:
        return self.file_path


def get_subject(p):
    return os.path.split(os.path.split(os.path.split(p)[0])[0])[1][4:]


def get_session(p):
    return os.path.split(os.path.split(p)[0])[1][4:]


def get_uid(p):
    return f"{get_subject(p)}_{get_session(p)}"


def get_candidate_info_list(folder, df, candidates: List[str]):
    candidate_info_list = []
    df = df.reset_index()  # make sure indexes pair with number of rows

    for _, row in df.iterrows():
        candidate = row["subject_id"]
        if candidate in candidates:
            append_candidate(folder, candidate_info_list, row)

    candidate_info_list.sort(reverse=True)

    return candidate_info_list


def append_candidate(folder, candidate_info_list, row):
    subject_str = row["subject_id"]
    session_str = row["session_id"]
    run_int = row["run_id"]
    suffix_str = row["suffix"]
    file_name = f"{subject_str}_{session_str}_run-{run_int}_{suffix_str}.nii.gz"
    file_path = os.path.join(folder, file_name)
    qu_motion_float = float(row["QU_motion"])
    candidate_info_list.append(
        CandidateInfoTuple(qu_motion_float, file_path, subject_str, session_str, run_int, suffix_str)
    )


# def get_subject_session_info(row, partial_loes_scores, anatomical_region):
#     subject_session_uid = row[1].strip()
#     pos = subject_session_uid.index("_")
#     session_str = subject_session_uid[pos + 1 :]
#     subject_str = row[0]
#     session = partial_loes_scores[subject_str][subject_session_uid]
#     if anatomical_region == "ParietoOccipitalWhiteMatter":
#         loes_score = session.parieto_occipital_white_matter.get_score()
#     elif anatomical_region == "all":
#         loes_score = session.loes_score
#     else:
#         assert False

#     return session_str, subject_session_uid, subject_str, loes_score


def z_normalize(image, mask=None):
    """Z-normalization (standardization) of image data"""
    if mask is not None:
        masked_data = image[mask > 0]
        mean = np.mean(masked_data)
        std = np.std(masked_data)
    else:
        mean = np.mean(image)
        std = np.std(image)

    if std == 0:
        return image - mean
    return (image - mean) / std

def resize_or_pad(image, target_shape=(260, 320, 320)):
    """Resize or pad image to target shape"""
    current_shape = image.shape
    
    # calculate padding or cropping needed for each dimension
    padded_image = np.zeros(target_shape, dtype=image.dtype)

    # calculate slice positions to center the image
    slices_in = []
    slices_out = []
    
    for i in range(3):
        if current_shape[i] <= target_shape[i]:
            # need to pad - take all of input, place in center of output
            start_out = (target_shape[i] - current_shape[i]) // 2
            slices_out.append(slice(start_out, start_out + current_shape[i]))
            slices_in.append(slice(None))
        else:
            # need to crop - take center of input, fill all of output
            start_in = (current_shape[i] - target_shape[i]) // 2
            slices_in.append(slice(start_in, start_in + target_shape[i]))
            slices_out.append(slice(None))

    padded_image[slices_out[0], slices_out[1], slices_out[2]] = image[slices_in[0], slices_in[1], slices_in[2]]

    return padded_image


def random_flip_lr(image, prob=0.5):
    """Random left-right flip"""
    if np.random.random() < prob:
        return np.flip(image, axis=0)  # Assuming first axis is left-right
    return image


def random_affine_transform(image, prob=0.8):
    """Simple random affine transformation using scipy"""
    if np.random.random() > prob:
        return image

    # Small random rotation (in degrees)
    angle = np.random.uniform(-5, 5)

    # Small random translation
    translation = [np.random.uniform(-2, 2) for _ in range(3)]

    # Apply rotation around center
    for axis in [(0, 1), (0, 2), (1, 2)]:
        if np.random.random() < 0.3:  # 30% chance for each axis pair
            image = ndimage.rotate(image, angle, axes=axis, reshape=False, order=1)

    # Apply translation
    image = ndimage.shift(image, translation, order=1)

    return image


class AutoQcMRIs:
    def __init__(self, candidate_info, is_val_set_bool):
        scan_path = candidate_info.path_to_file

        # Load NIfTI file using nibabel instead of TorchIO
        nii_img = nib.load(scan_path)
        image_data = nii_img.get_fdata()

        # Convert to float32 and ensure it's a numpy array
        image_data = np.array(image_data, dtype=np.float32)
        
        # Resize or pad to target shape
        image_data = resize_or_pad(image_data, target_shape=(260, 320, 320))

        # Z-normalization (equivalent to tio.ZNormalization)
        image_data = z_normalize(image_data)

        # Apply augmentations only for training
        if not is_val_set_bool:
            # Random left-right flip (equivalent to tio.RandomFlip(axes='LR'))
            image_data = random_flip_lr(image_data)

            # Random affine or elastic deformation (simplified version)
            if np.random.random() < 0.8:
                image_data = random_affine_transform(image_data)
            # Note: Elastic deformation is more complex without TorchIO

        # Convert to torch tensor and add channel dimension
        self.mri_image_tensor = torch.from_numpy(image_data.copy()).unsqueeze(
            0
        )  # Add channel dim
        self.subject_session_uid = candidate_info

    def get_raw_candidate(self):
        return self.mri_image_tensor


@functools.lru_cache(1, typed=True)
def get_auto_qc_mris(candidate_info, is_val_set_bool):
    return AutoQcMRIs(candidate_info, is_val_set_bool)


@raw_cache.memoize(typed=True)
def get_mri_raw_candidate(subject_session_uid, is_val_set_bool):
    auto_qc_mris = get_auto_qc_mris(subject_session_uid, is_val_set_bool)
    mri_image_tensor = auto_qc_mris.get_raw_candidate()

    return mri_image_tensor


class AutoQcDataset(Dataset):
    def __init__(
        self,
        folder,
        subjects: List[str],
        df,
        output_df,
        is_val_set_bool=None,
        subject=None,
        sortby_str="random"
    ):
        self.is_val_set_bool = is_val_set_bool
        self.candidateInfo_list = copy.copy(
            get_candidate_info_list(folder, df, subjects)
        )

        if subject:
            self.candidateInfo_list = [
                x for x in self.candidateInfo_list if x.subject_str == subject
            ]

        if sortby_str == "random":
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == "QU_motion":
            self.candidateInfo_list.sort(key=lambda x: x.qu_motion_float)
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        log.info(
            "{!r}: {} {} samples".format(
                self,
                len(self.candidateInfo_list),
                "validation" if is_val_set_bool else "training",
            )
        )
        if output_df is not None:
            for candidate_info in self.candidateInfo_list:
                row_location = (
                    df["subject_id"] == candidate_info.subject
                ) & (df["session_id"] == candidate_info.session_str)
                output_df.loc[row_location, "training"] = 0 if is_val_set_bool else 1
                output_df.loc[row_location, "validation"] = 1 if is_val_set_bool else 0

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        candidate_info = self.candidateInfo_list[ndx]
        candidate_a = get_mri_raw_candidate(candidate_info, self.is_val_set_bool)
        candidate_t = candidate_a.to(torch.float32)

        qu_motion = candidate_info.qu_motion_float
        qu_motion_t = torch.tensor(qu_motion, dtype=torch.float32)

        return (
            candidate_t,
            qu_motion_t,
            candidate_info.subject_str,
            candidate_info.session_str,
        )
