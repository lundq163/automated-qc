"""QU_motion Prediction Module

This module provides functionality for predicting QU_motion scores from brain MRI scans
using deep learning models (Regressor or AlexNet3D). It handles model loading, image preprocessing,
inference, and validation against known scores.

Key Features:
    - Load pre-trained Regressor or AlexNet3D models for QU_motion score prediction
    - Process NIfTI brain MRI files with z-normalization
    - Batch predictions on directories of MRI scans
    - Validation and comparison with ground truth scores
    - Scatter plot visualization of predictions vs actual scores
    - RMSE and correlation coefficient computation

Usage:
    python make_predictions.py <model_file> <output_csv> <nifti_dir> [options]
    
Example:
    python make_predictions.py model.pt predictions.csv /data/mri_scans \
        --model_type regressor --device gpu --validation_csv_file_path validation.csv
"""

import logging
import sys

import pandas as pd
import statistics
import torch
import matplotlib.pyplot as plt
import numpy as np

import glob
import os
import nibabel as nib
import argparse

from data_sets.dsets import resize_or_pad, z_normalize
from models.torchmodels import AlexNet3D
from models.regressor import get_regressor_model

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


def load_model(model_name, model_save_location, device="cpu"):
    """Load a pre-trained QU_motion model.

    Args:
        model_name (str): Model architecture type ('regressor' or 'alexnet')
        model_save_location (str): Path to saved model checkpoint (.pt file)
        device (str): Device to load model on ('cpu' or 'cuda')

    Returns:
        torch.nn.Module: Loaded model in evaluation mode

    Raises:
        FileNotFoundError: If model file doesn't exist
        PermissionError: If model file is not readable
        ValueError: If model file has invalid format
        RuntimeError: If model architecture doesn't match checkpoint
    """
    # Check if model file exists
    if not os.path.exists(model_save_location):
        raise FileNotFoundError(f"Model file not found at: {model_save_location}")

    # Check if the file is readable
    if not os.access(model_save_location, os.R_OK):
        raise PermissionError(f"Cannot read model file at: {model_save_location}")

    # Initialize model based on type
    if model_name.lower() == "regressor":
        model = get_regressor_model()
        log.info("Using Regressor")
    else:
        model = AlexNet3D(4608)
        log.info("Using AlexNet3D")

    model.to(device)

    # Load model with error handling
    try:
        state_dict = torch.load(
            model_save_location, map_location=device, weights_only=True
        )
        model.load_state_dict(state_dict)
        log.info(f"Successfully loaded model from {model_save_location}")
    except torch.serialization.pickle.UnpicklingError as e:
        raise ValueError(f"Invalid model file format at {model_save_location}: {e}")
    except RuntimeError as e:
        raise RuntimeError(
            f"Model architecture mismatch or corrupted file at {model_save_location}: {e}"
        )
    except Exception as e:
        raise Exception(f"Failed to load model from {model_save_location}: {e}")

    model.eval()
    return model


def predict(row, data_folder):
    """Generate input tensor for a single subject/session from CSV row.

    Args:
        row (pd.Series): DataFrame row with subject and session IDs
        data_folder (str): Directory containing preprocessed NIfTI files

    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input
    """
    subject = row["subject_id"]
    session = row["session_id"]
    run = row["run_id"]
    suffix = row["suffix"]
    scan_path = (
        f"{data_folder}/{subject}_{session}_run-{run}_{suffix}.nii.gz"
    )
    scan_image_tensor = get_image_tensor(scan_path)
    value = scan_image_tensor.unsqueeze(0)

    return value


def get_image_tensor(mri_path):
    """Load and preprocess a NIfTI MRI scan for model input.

    Args:
        mri_path (str): Path to NIfTI file (.nii.gz format)

    Returns:
        torch.Tensor: Preprocessed 4D tensor (1, C, H, W, D) with z-normalization applied
    """
    # Load NIfTI file using nibabel
    nii_img = nib.load(mri_path)
    image_data = nii_img.get_fdata()

    # Convert to float32 and ensure it's a numpy array
    image_data = np.array(image_data, dtype=np.float32)
    
    # Resize or pad to target shape
    image_data = resize_or_pad(image_data, target_shape=(260, 320, 320))

    # Z-normalization (equivalent to tio.ZNormalization with mean masking)
    image_data = z_normalize(image_data)

    # Convert to torch tensor and add channel dimension
    mri_image_tensor = torch.from_numpy(image_data).unsqueeze(0)  # Add channel dim

    # Move to CPU (though it's already on CPU)
    input_g = mri_image_tensor.to("cpu", non_blocking=True)

    return input_g


def compute_rmse(predictions, actuals):
    """Calculate Root Mean Square Error between predictions and actual values.

    Args:
        predictions (list): Predicted QU_motion scores
        actuals (list): Actual/ground truth QU_motion scores

    Returns:
        float: RMSE value
    """
    predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
    actuals_tensor = torch.tensor(actuals, dtype=torch.float32)
    mse = torch.nn.functional.mse_loss(predictions_tensor, actuals_tensor)
    return torch.sqrt(mse).item()


def get_validation_info(
    model_type, model_save_location, input_csv_location, val_subjects, data_folder
):
    """Generate predictions for validation subjects.

    Args:
        model_type (str): Model architecture ('regressor' or 'alexnet')
        model_save_location (str): Path to model checkpoint
        input_csv_location (str): CSV with subject metadata
        val_subjects (list): List of validation subject IDs
        data_folder (str): Directory with preprocessed MRI files

    Returns:
        tuple: (subjects, sessions, actual_scores, predicted_scores)
    """
    model = load_model(model_type, model_save_location, device="cpu")

    df = pd.read_csv(input_csv_location)
    validation_rows = df[df["subject_id"].isin(val_subjects)]
    output_df = validation_rows.copy()
    subjects = list(output_df["subject_id"])
    sessions = list(output_df["session_id"])
    actual_scores = list(output_df["QU_motion"])
    with torch.no_grad():
        inputs = list(output_df.apply(predict, axis=1, args=(data_folder,)))

        predictions = [model(input) for input in inputs]
        predict_vals = [p[0].item() for p in predictions]

        return subjects, sessions, actual_scores, predict_vals


def compute_standardized_rmse(actual_scores, predict_vals):
    """Compute RMSE normalized by standard deviation of actual scores.

    Args:
        actual_scores (list): Ground truth QU_motion scores
        predict_vals (list): Predicted QU_motion scores

    Returns:
        float: Standardized RMSE (RMSE / σ_actual)
    """
    rmse = compute_rmse(predict_vals, actual_scores)
    sigma = statistics.stdev(actual_scores)
    standardized_rmse = rmse / sigma

    return standardized_rmse


def create_correlation_coefficient(actual_vals, predicted_vals):
    """Calculate Pearson correlation coefficient between actual and predicted values.

    Args:
        actual_vals (list): Actual QU_motion scores
        predicted_vals (list): Predicted QU_motion scores

    Returns:
        float: Pearson correlation coefficient (r value)
    """
    x = np.array(actual_vals)
    y = np.array(predicted_vals)

    correlation_matrix = np.corrcoef(x, y)
    correlation_coefficient = correlation_matrix[0, 1]

    return correlation_coefficient


def create_scatter_plot(actual_vals, predicted_vals, output_file):
    """Generate scatter plot comparing predicted vs actual QU_motion scores.

    Creates a scatter plot with:
    - Points colored by prediction error magnitude
    - Perfect prediction diagonal line
    - Equal aspect ratio for fair comparison

    Args:
        actual_vals (list): Ground truth QU_motion scores
        predicted_vals (list): Model predicted QU_motion scores
        output_file (str): Path to save plot image
    """
    _, ax = plt.subplots(figsize=(8, 6))

    # Color by prediction error
    errors = np.abs(np.array(actual_vals) - np.array(predicted_vals))
    scatter = ax.scatter(
        actual_vals,
        predicted_vals,
        s=30,
        c=errors,
        cmap=plt.cm.Reds,
        alpha=0.7,
        zorder=10,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Prediction Error", fontsize=11)

    # Perfect prediction line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.8, linewidth=2)

    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Labels
    ax.set_xlabel("Actual QU_motion score")
    ax.set_ylabel("Predicted QU_motion score")
    ax.set_title("QU_motion score prediction")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def get_predicted_value(row, subjects, sessions, runs, suffixes, predict_vals):
    """Match predicted value to subject/session/run/suffix combination.

    Args:
        row (pd.Series): DataFrame row with subject and session IDs
        subjects (list): List of subject IDs
        sessions (list): List of session IDs
        runs (list): List of run IDs
        suffixes (list): List of suffix IDs
        predict_vals (list): Corresponding predicted values

    Returns:
        float: Predicted value for the subject/session/run/suffix, or NaN if not found
    """
    zipped_data = zip(subjects, sessions, runs, suffixes, predict_vals)

    for subject, session, run, suffix, predict_val in zipped_data:
        if (
            row["subject_id"] == subject
            and row["session_id"] == session
            and row["run_id"] == run
            and row["suffix"] == suffix
        ):
            return predict_val
    return np.nan


def add_predicted_values(subjects, sessions, runs, suffixes, predict_vals, input_csv_location):
    """Add predicted QU_motion scores to existing CSV data.

    Args:
        subjects (list): Subject IDs with predictions
        sessions (list): Session IDs with predictions
        runs (list): Run IDs with predictions
        suffixes (list): Suffix IDs with predictions
        predict_vals (list): Predicted QU_motion scores
        input_csv_location (str): Path to input CSV file

    Returns:
        pd.DataFrame: Input data with added 'predicted_qu_motion_score' column
    """
    input_df = pd.read_csv(input_csv_location)
    output_df = input_df.copy()
    output_df["predicted_qu_motion_score"] = output_df.apply(
        get_predicted_value, axis=1, args=(subjects, sessions, runs, suffixes, predict_vals)
    )

    return output_df


def get_files_by_pattern(directory, pattern):
    """
    Retrieves all files in a directory matching a specified filename pattern.

    Args:
        directory (str): The path to the directory to search.
        pattern (str): The filename pattern to match (e.g., "*.txt", "image_*.png").

    Returns:
        list: A list of file paths that match the pattern.
    """
    search_pattern = os.path.join(directory, pattern)
    files = glob.glob(search_pattern)
    return files


def get_filename_from_path(file_path):
    """
    Extracts the filename from a given file path.

    Args:
        file_path: The path to the file.

    Returns:
        The filename, or None if the path is invalid.
    """
    return os.path.basename(file_path)


def make_predictions_on_folder(directory_path, file_pattern, model):
    """Generate QU_motion predictions for all matching MRI files in a directory.

    
    Processes all NIfTI files matching the pattern, extracting subject/session/run/suffix IDs
    from filenames (expected format: sub-<subject>_ses-<session>_run-<run>_<suffix>.nii.gz).

    Args:
        directory_path (str): Directory containing MRI files
        file_pattern (str): Glob pattern for MRI files
        model (torch.nn.Module): Loaded model for prediction

    Returns:
        pd.DataFrame: DataFrame with columns: subject_id, session_id, run_id, suffix, predicted_score
    """
    matching_files = get_files_by_pattern(directory_path, file_pattern)

    df = pd.DataFrame(
        {
            "subject_id": [],
            "session_id": [],
            "run_id": [],
            "suffix": [],
            "predicted_score": [],
        }
    )
    if matching_files:
        for file_path in matching_files:
            image_tensor = get_image_tensor(file_path)
            file_name = get_filename_from_path(file_path)
            parts = file_name.split("_")
            subject_id = parts[0]
            session_id = parts[1]
            run_id = parts[2]
            suffix = parts[3]
            with torch.no_grad():
                unsqueezed_image_tensor = image_tensor.unsqueeze(0)
                prediction = model(unsqueezed_image_tensor)
                prediction_p = prediction.item()
                new_row = pd.DataFrame(
                    {
                        "subject_id": [subject_id],
                        "session_id": [session_id],
                        "run_id": [run_id],
                        "suffix": [suffix],
                        "predicted_score": [prediction_p],
                    }
                )
                df = pd.concat([df, new_row], ignore_index=True)
    else:
        print("No files found matching the pattern.")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions on input files.")
    parser.add_argument("model_file_path", help="Path to the model file")
    parser.add_argument("input_csv_file_path", help="Path to the CSV file")
    parser.add_argument(
        "nifti_directory_path", help="Path to folder containing NIFTI files"
    )
    parser.add_argument(
        "--file_pattern",
        default="*.nii.gz",
        help="The pattern of the NIFTI files, such as *.nii.gz",
    )
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["gpu", "cpu"],
        help="Choose the device (gpu or cpu)",
    )
    parser.add_argument(
        "--model_type",
        default="regressor",
        choices=["regressor", "alexnet"],
        help="Choose the model architecture (regressor or alexnet)",
    )
    parser.add_argument(
        "--validation_csv_file_path",
        help="For validation of known QU_motion scores, the path to the CSV containing those scores.",
    )
    parser.add_argument(
        "--scatter_plot_file_path", help="File path to output scatter plot"
    )
    args = parser.parse_args()
    model_save_location = sys.argv[1]
    csv_file_name = sys.argv[2]
    directory_path = sys.argv[3]
    file_pattern = args.file_pattern
    model = load_model(args.model_type, model_save_location, device=args.device)
    df = make_predictions_on_folder(directory_path, file_pattern, model)
    df.to_csv(csv_file_name, index=False)
    if args.validation_csv_file_path:
        expected_df = pd.read_csv(args.validation_csv_file_path)
        expected_df = pd.read_csv()
        expected_validation_df = expected_df[expected_df["validation"] == 1]
        merged_df = pd.merge(
            df,
            expected_validation_df,
            on=["subject_id", "session_id", "run_id", "suffix"],
            how="inner",
        )
        actual_scores = list(merged_df["qu-motion-score"])
        predict_vals = list(merged_df["predicted_score"])
        standardized_rmse = compute_standardized_rmse(actual_scores, predict_vals)
        print(f"standardized_rmse: {standardized_rmse}")
        create_scatter_plot(actual_scores, predict_vals, args.scatter_plot_file_path)
