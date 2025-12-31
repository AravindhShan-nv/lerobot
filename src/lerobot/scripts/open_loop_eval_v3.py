"""Open-loop evaluation script for GR00T N1.6 models on LeRobot v3 datasets.

This script evaluates GR00T N1.6 policies on LeRobot v3 datasets from Hugging Face Hub.
It supports:
- Loading v3 datasets directly from Hub
- Evaluating model predictions with open-loop action prediction
- Computing metrics (MSE, MAE) and generating trajectory comparison plots

Alignment with Isaac-GR00T (gr00t/eval/open_loop_eval.py):
===========================================================
This script is designed to match the evaluation pattern from Isaac-GR00T:
- Uses EmbodimentTag from the policy's processor (detected during policy init)
- Uses processor's modality_configs for action/state keys (not hardcoded)
- Follows the same inference pattern: step through episode at action_horizon intervals
- Actions are unnormalized via processor.decode_action()

Important Notes on Embodiment and Action Decoding:
=================================================
1. The policy determines its embodiment tag from the processor's available configs
   (e.g., behavior_r1_pro, gr1, robocasa_panda_omron). The --embodiment-tag CLI arg
   is informational; the policy's actual embodiment is used for correct decoding.

2. Action unnormalization is handled by processor.decode_action() which:
   - Splits the model output by joint groups based on modality_configs
   - Unnormalizes using stored statistics (min/max or mean/std)
   - Optionally converts relative->absolute actions (requires proper state keys)

3. For relative->absolute conversion, the processor expects state organized by
   specific keys (e.g., arm_left_qpos, trunk_qpos for behavior_r1_pro). LeRobot
   datasets have flattened state, so relative actions may remain relative.
   This is fine for comparison if ground truth is also in the same space.

4. Action dimensions: The model may output more dimensions than the dataset
   (e.g., 23 dims for behavior_r1_pro vs 6 dims in some datasets). The script
   truncates to dataset dimensions for metric comparison.

Usage examples:
--------------
- Visualize dataset trajectories only (no model evaluation):
    python open_loop_eval_v3.py \
        --dataset-repo-id=izuluaga/finish_sandwich \
        --episode-ids=0 \
        --visualize-only=True \
        --save-dir=./outputs

- Evaluate multiple episodes:
    python open_loop_eval_v3.py \
        --dataset-repo-id=izuluaga/finish_sandwich \
        --policy-repo-id=nvkartik/gr00t_n1d6-finish_sandwich-relative-action-true-tune-30k \
        --episode-ids=10 11 13 \
        --save-dir=./outputs \
        --action-horizon=16 \
        --steps=400 \
        --inference-interval=2 \
        --action-offset=7

- Evaluate with specific embodiment tag:
    python open_loop_eval_v3.py \
        --dataset-repo-id=HuggingFaceVLA/libero \
        --policy-repo-id=nvkartik/gr00t_n1d6-libero-rel-action-false-tune-false-30k \
        --episode-ids=0 \
        --save-dir=./outputs \
        --action-horizon=16 \
        --steps=400 \
        --inference-interval=2 \
        --action-offset=7
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from matplotlib import pyplot as plt
# LeRobot imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class
from lerobot.policies.gr00t_n1d6.utils import EmbodimentTag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_trajectory_results(
    state_joints_across_time: np.ndarray,
    gt_action_across_time: np.ndarray,
    pred_action_across_time: np.ndarray | None,
    traj_id: int,
    state_keys: list[str],
    action_keys: list[str],
    action_horizon: int,
    save_plot_path: str,
) -> None:
    """
    Plot and save trajectory results comparing ground truth and predicted actions.

    Args:
        state_joints_across_time: Array of state joints over time
        gt_action_across_time: Ground truth actions over time
        pred_action_across_time: Predicted actions over time (None for visualize-only mode)
        traj_id: Trajectory ID
        state_keys: List of state modality keys
        action_keys: List of action modality keys
        action_horizon: Action horizon used for inference (not used if pred_action_across_time is None)
        save_plot_path: Path to save the plot
    """
    actual_steps = len(gt_action_across_time)
    action_dim = gt_action_across_time.shape[1]

    indices_to_plot = list(range(action_dim))

    num_plots = len(indices_to_plot)
    if num_plots == 0:
        logger.warning("No valid indices to plot")
        return

    # Always plot and save
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(8, 4 * num_plots))

    # Handle case where there's only one subplot
    if num_plots == 1:
        axes = [axes]

    # Add a global title showing the modality keys (format similar to SO100 example)
    all_modalities = state_keys + action_keys
    fig.suptitle(
        f"Trajectory {traj_id} - Modalities: {', '.join(all_modalities)}",
        fontsize=16,
        color="blue",
    )

    for plot_idx, action_idx in enumerate(indices_to_plot):
        ax = axes[plot_idx]

        # The dimensions of state_joints and action are the same
        # only when the robot uses actions directly as joint commands.
        # Therefore, do not plot them if this is not the case.
        if state_joints_across_time.shape == gt_action_across_time.shape:
            ax.plot(state_joints_across_time[:, action_idx], label="state joints")
        ax.plot(gt_action_across_time[:, action_idx], label="gt action")

        # Only plot predictions if available
        if pred_action_across_time is not None:
            ax.plot(pred_action_across_time[:, action_idx], label="pred action")
            # put a dot every ACTION_HORIZON (only when predictions are available)
            for j in range(0, actual_steps, action_horizon):
                if j == 0:
                    ax.plot(j, gt_action_across_time[j, action_idx], "ro", label="inference point")
                else:
                    ax.plot(j, gt_action_across_time[j, action_idx], "ro")

        ax.set_title(f"Action {action_idx}")
        ax.legend()

    plt.tight_layout()

    # Create filename with trajectory ID
    Path(save_plot_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_plot_path)

    plt.close()  # Close the figure to free memory


def get_episode_frames(dataset: LeRobotDataset, episode_id: int) -> list[dict[str, Any]]:
    """
    Get all frames for a specific episode from LeRobotDataset.

    Args:
        dataset: LeRobotDataset instance
        episode_id: Episode index

    Returns:
        List of frame dictionaries (processed through dataset.__getitem__)
    """
    dataset._ensure_hf_dataset_loaded()

    # First, find all frame indices belonging to this episode
    episode_indices = []
    for idx in range(len(dataset.hf_dataset)):
        if dataset.hf_dataset[idx]["episode_index"] == episode_id:
            episode_indices.append(idx)

    # Now get processed frames using dataset.__getitem__
    episode_frames = []
    for idx in episode_indices:
        try:
            frame = dataset[idx]
            episode_frames.append(frame)
        except Exception as e:
            logger.warning(f"Error loading frame {idx} from episode {episode_id}: {e}")
            continue

    logger.info(f"Found {len(episode_frames)} frames for episode {episode_id}")
    return episode_frames


def visualize_dataset_trajectory(
    dataset: LeRobotDataset,
    episode_id: int,
    steps: int = 200,
    save_plot_path: str | None = None,
) -> None:
    """
    Visualize ground truth trajectories from the dataset without model evaluation.

    Args:
        dataset: LeRobotDataset instance
        episode_id: Episode index to visualize
        steps: Maximum number of steps to visualize
        save_plot_path: Path to save the plot
    """
    # Get all frames for this episode
    episode_frames = get_episode_frames(dataset, episode_id)
    if len(episode_frames) == 0:
        raise ValueError(f"No frames found for episode {episode_id}")

    traj_length = len(episode_frames)
    actual_steps = min(steps, traj_length)
    logger.info(f"Visualizing {actual_steps} steps (requested: {steps}, trajectory length: {traj_length})")

    state_key = "observation.state"
    action_key = "action"

    # Extract state and action keys
    state_keys = ["state"]
    action_keys = ["action"]

    # Get state and action dimensions
    state_dim = (
        dataset.meta.features.get(state_key, {}).get("shape", [0])[0]
        if state_key in dataset.meta.features
        else 0
    )
    action_dim = (
        dataset.meta.features.get(action_key, {}).get("shape", [0])[0]
        if action_key in dataset.meta.features
        else 0
    )

    logger.info(f"State keys: {state_keys}, Action keys: {action_keys}")
    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Extract ground truth trajectories
    state_joints_across_time = []
    gt_action_across_time = []

    # Infer dimensions from first frame if not available from metadata
    inferred_state_dim = state_dim
    inferred_action_dim = action_dim

    for step_count in range(actual_steps):
        if step_count >= len(episode_frames):
            break

        frame = episode_frames[step_count]

        # Extract state
        if state_key in frame:
            state = frame[state_key]
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            if inferred_state_dim == 0 and state.size > 0:
                inferred_state_dim = state.size if state.ndim == 0 else state.shape[-1]
            state_joints_across_time.append(state.flatten() if state.ndim > 1 else state)
        elif inferred_state_dim > 0:
            # Use zeros if state not available but we know the dimension
            state_joints_across_time.append(np.zeros(inferred_state_dim))

        # Extract action
        if action_key in frame:
            action = frame[action_key]
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            if inferred_action_dim == 0 and action.size > 0:
                inferred_action_dim = action.size if action.ndim == 0 else action.shape[-1]
            gt_action_across_time.append(action.flatten() if action.ndim > 1 else action)
        elif inferred_action_dim > 0:
            # Use zeros if action not available but we know the dimension
            gt_action_across_time.append(np.zeros(inferred_action_dim))

    # Convert to numpy arrays
    state_joints_across_time = np.array(state_joints_across_time)
    gt_action_across_time = np.array(gt_action_across_time)

    # Plot trajectory results (only ground truth, no predictions)
    plot_trajectory_results(
        state_joints_across_time=state_joints_across_time,
        gt_action_across_time=gt_action_across_time,
        pred_action_across_time=None,  # No predictions in visualize-only mode
        traj_id=episode_id,
        state_keys=state_keys,
        action_keys=action_keys,
        action_horizon=1,  # Not used when pred_action_across_time is None
        save_plot_path=save_plot_path,
    )


def parse_observation_gr00t(obs: dict[str, Any], modality_configs: dict[str, Any]) -> dict[str, Any]:
    """
    Parse observation into GR00T format (aligned with Isaac-GR00T's parse_observation_gr00t).

    Args:
        obs: Observation dict with keys like 'video.{key}', 'state.{key}', 'task'
        modality_configs: Modality configs from processor

    Returns:
        Nested observation dict: {video: {}, state: {}, language: {}}
    """
    new_obs = {}
    for modality in ["video", "state", "language"]:
        new_obs[modality] = {}
        modality_config = modality_configs.get(modality)
        if modality_config is None:
            continue
        for key in modality_config.modality_keys:
            parsed_key = key if modality == "language" else f"{modality}.{key}"
            if parsed_key in obs:
                arr = obs[parsed_key]
                # Add batch dimension
                if isinstance(arr, str):
                    new_obs[modality][key] = [[arr]]
                else:
                    new_obs[modality][key] = arr[None, :]
    return new_obs


def parse_action_gr00t(action: dict[str, Any]) -> dict[str, Any]:
    """
    Parse action from GR00T format (aligned with Isaac-GR00T's parse_action_gr00t).

    Args:
        action: Action dict from policy output

    Returns:
        Dict with unbatched actions with 'action.' prefix
    """
    # Unbatch and add prefix
    return {f"action.{key}": action[key][0] for key in action}


def pad_state_to_checkpoint_dim(state: np.ndarray, checkpoint_max_state_dim: int) -> np.ndarray:
    """Pad state array to match checkpoint's expected dimension.
    
    This ensures state has the correct dimension expected by the model checkpoint.
    The processor (Gr00tN1d6Processor.__call__) does this padding at lines 871-877.
    
    Args:
        state: State array of shape (D,) or (T, D)
        checkpoint_max_state_dim: Expected state dimension from checkpoint
        
    Returns:
        Padded state array with last dimension = checkpoint_max_state_dim
    """
    if state.ndim == 1:
        # (D,) -> pad to (checkpoint_max_state_dim,)
        if state.shape[0] < checkpoint_max_state_dim:
            padding = np.zeros(checkpoint_max_state_dim - state.shape[0], dtype=state.dtype)
            return np.concatenate([state, padding])
        return state[:checkpoint_max_state_dim]
    elif state.ndim == 2:
        # (T, D) -> pad to (T, checkpoint_max_state_dim)
        if state.shape[1] < checkpoint_max_state_dim:
            padding = np.zeros((state.shape[0], checkpoint_max_state_dim - state.shape[1]), dtype=state.dtype)
            return np.concatenate([state, padding], axis=1)
        return state[:, :checkpoint_max_state_dim]
    else:
        raise ValueError(f"Unexpected state shape: {state.shape}")


def prepare_inference_batch(
    frame: dict[str, Any],
    camera_keys: list[str],
    state_key: str,
    language_key: str,
    processor: Any,
    embodiment_tag: Any,
    checkpoint_max_state_dim: int,
    device: torch.device,
) -> dict[str, Any]:
    """Prepare batch for inference using processor methods.
    
    This function uses the processor's _apply_vlm_processing() method for VLM content
    creation, ensuring consistency between training and inference preprocessing.
    
    Args:
        frame: Single frame dict from LeRobotDataset
        camera_keys: List of camera keys (e.g., ['observation.images.top'])
        state_key: Key for state data (e.g., 'observation.state')
        language_key: Key for language/task data (e.g., 'task')
        processor: Gr00tN1d6Processor instance from the policy
        embodiment_tag: EmbodimentTag for the policy
        checkpoint_max_state_dim: Expected state dimension from checkpoint
        device: Device to place tensors on
        
    Returns:
        Dictionary with keys: vlm_content, state, raw_state, embodiment_id
    """
    from lerobot.policies.gr00t_n1d6.utils import EMBODIMENT_TAG_TO_PROJECTOR_INDEX
    
    policy_batch = {}
    
    # =========================================================================
    # 1. Extract and format images for processor's _apply_vlm_processing
    #    Expected format: [T, C, H, W] numpy array (uint8)
    # =========================================================================
    images_list = []
    for cam_key in camera_keys:
        if cam_key in frame:
            img = frame[cam_key]
            # Convert to numpy if tensor
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            
            # Handle different image formats
            # LeRobot stores images as (C, H, W) - need to ensure this format
            if img.ndim == 3:
                # Could be (C, H, W) or (H, W, C)
                if img.shape[0] in [1, 3, 4]:  # Likely (C, H, W) format
                    pass  # Already in correct format
                else:
                    # Assume (H, W, C), convert to (C, H, W)
                    img = np.transpose(img, (2, 0, 1))
            
            # Ensure uint8 format
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
            
            images_list.append(img)
    
    # Stack images to [T, C, H, W] format (T=number of camera views for single timestep)
    if images_list:
        stacked_images = np.stack(images_list, axis=0)  # [T, C, H, W]
    else:
        raise ValueError(f"No images found in frame. Camera keys: {camera_keys}")
    
    # =========================================================================
    # 2. Extract language/task text
    # =========================================================================
    task_text = "complete the task"  # Default
    if language_key in frame:
        task_data = frame[language_key]
        if isinstance(task_data, str):
            task_text = task_data
        elif isinstance(task_data, (list, tuple)) and len(task_data) > 0:
            task_text = task_data[0] if isinstance(task_data[0], str) else str(task_data[0])
    
    # =========================================================================
    # 3. Use processor's _apply_vlm_processing for VLM content
    #    This matches processor_gr00t_n1d6.py lines 747-782
    # =========================================================================
    vlm_result = processor._apply_vlm_processing(stacked_images, task_text)
    policy_batch["vlm_content"] = vlm_result["vlm_content"]
    
    # =========================================================================
    # 4. Extract and prepare state with padding
    #    This matches processor_gr00t_n1d6.py lines 871-877
    # =========================================================================
    if state_key in frame:
        state = frame[state_key]
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        
        # Flatten to 1D if needed
        if state.ndim > 1:
            state = state.flatten()
        
        # Store raw state for relative->absolute action conversion
        policy_batch["raw_state"] = {"state": state}
        
        # Pad state to checkpoint's expected dimension
        padded_state = pad_state_to_checkpoint_dim(state, checkpoint_max_state_dim)
        
        # Convert to tensor with batch dimension: (D,) -> (1, D)
        state_tensor = torch.from_numpy(padded_state).unsqueeze(0).to(device).float()
        policy_batch["state"] = state_tensor
    else:
        raise ValueError(f"State key '{state_key}' not found in frame")
    
    # =========================================================================
    # 5. Set embodiment_id
    # =========================================================================
    embodiment_id = EMBODIMENT_TAG_TO_PROJECTOR_INDEX.get(
        embodiment_tag.value, 10  # Default to new_embodiment=10
    )
    policy_batch["embodiment_id"] = torch.tensor([embodiment_id], device=device, dtype=torch.long)
    
    return policy_batch


def evaluate_single_trajectory(
    policy: Any,
    dataset: LeRobotDataset,
    episode_id: int,
    embodiment_tag: EmbodimentTag,
    modality_configs: dict[str, Any],
    steps: int = 200,
    action_horizon: int = 16,
    inference_interval: int | None = None,
    save_plot_path: str | None = None,
    action_offset: int = 0,
) -> tuple[float, float]:
    """
    Evaluate a single trajectory/episode (aligned with Isaac-GR00T's evaluate_single_trajectory).

    Args:
        policy: GR00T N1.6 policy instance
        dataset: LeRobotDataset instance
        episode_id: Episode index to evaluate
        embodiment_tag: EmbodimentTag for the policy (e.g., EmbodimentTag.NEW_EMBODIMENT)
        modality_configs: Modality configs from processor for this embodiment
        steps: Maximum number of steps to evaluate
        action_horizon: Action horizon for inference (number of steps predicted at once)
        inference_interval: Interval between inference points. If None, uses action_horizon.
        save_plot_path: Path to save the plot
        action_offset: Offset into model's action output for alignment with dataset.
            E.g., for behavior_r1_pro outputting [base(3), torso, arm...], use offset=3
            to skip base dims when comparing against a 6-DOF arm dataset.

    Returns:
        Tuple of (MSE, MAE) metrics
    """
    # Use action_horizon as default inference interval if not specified (matches Isaac-GR00T)
    if inference_interval is None:
        inference_interval = action_horizon

    # Get all frames for this episode
    episode_frames = get_episode_frames(dataset, episode_id)
    if len(episode_frames) == 0:
        raise ValueError(f"No frames found for episode {episode_id}")

    traj_length = len(episode_frames)
    actual_steps = min(steps, traj_length)
    logger.info(f"Using {actual_steps} steps (requested: {steps}, trajectory length: {traj_length})")

    # Pre-allocate arrays to handle overlapping predictions correctly
    pred_action_across_time = []

    camera_keys = dataset.meta.camera_keys
    state_key = "observation.state"
    action_key = "action"
    language_key = "task"

    # Extract state and action keys from modality_configs (aligned with Isaac-GR00T)
    state_keys = modality_configs.get("state", {}).modality_keys if "state" in modality_configs else ["state"]
    action_keys = (
        modality_configs.get("action", {}).modality_keys if "action" in modality_configs else ["action"]
    )

    # Get state and action dimensions from dataset features
    state_dim = (
        dataset.meta.features.get(state_key, {}).get("shape", [0])[0]
        if state_key in dataset.meta.features
        else 0
    )
    action_dim = (
        dataset.meta.features.get(action_key, {}).get("shape", [0])[0]
        if action_key in dataset.meta.features
        else 0
    )

    logger.info(f"Embodiment tag: {embodiment_tag.value}")
    logger.info(f"State keys (from modality_configs): {state_keys}")
    logger.info(f"Action keys (from modality_configs): {action_keys}")
    logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")
    logger.info(f"Inference interval: {inference_interval} steps, Action horizon: {action_horizon} steps")

    # Step through episode at inference_interval intervals
    for step_count in range(0, actual_steps, inference_interval):
        if step_count >= len(episode_frames):
            break

        logger.info(f"Inferencing at step: {step_count}")

        # Get current frame
        frame = episode_frames[step_count]

        # Prepare batch for inference using processor methods
        # This uses processor's _apply_vlm_processing for VLM content (lines 747-782)
        # and pad_state_to_checkpoint_dim for state padding (matches lines 871-877)
        policy_batch = prepare_inference_batch(
            frame=frame,
            camera_keys=camera_keys,
            state_key=state_key,
            language_key=language_key,
            processor=policy._processor,
            embodiment_tag=embodiment_tag,
            checkpoint_max_state_dim=policy._checkpoint_max_state_dim,
            device=policy.device,
        )
        
        # Log warning about state structure on first step
        if step_count == 0:
            state_modality_keys = (
                modality_configs.get("state", {}).modality_keys
                if "state" in modality_configs
                else ["state"]
            )
            if len(state_modality_keys) > 1:
                logger.warning(
                    f"Model expects {len(state_modality_keys)} state keys: {state_modality_keys[:5]}... "
                    f"LeRobot provides flat state. Relative->absolute conversion may be skipped."
                )

        # Run inference
        try:
            with torch.no_grad():
                # Use predict_action_chunk for action prediction
                action_chunk = policy.predict_action_chunk(policy_batch)
                # action_chunk shape: (B, action_horizon, model_action_dim)
                action_chunk_np = action_chunk.cpu().numpy()[0]  # Remove batch dimension

                # Log dimensions on first step for debugging
                if step_count == 0:
                    model_action_dim = action_chunk_np.shape[1]
                    logger.info(
                        f"Model output action dim: {model_action_dim}, Dataset action dim: {action_dim}"
                    )
                    logger.info(f"Action chunk shape: {action_chunk_np.shape}")
                    logger.info(
                        f"Action chunk first timestep sample: {action_chunk_np[0, : min(6, model_action_dim)]}"
                    )

                # Truncate/slice to dataset action_dim if model outputs more dimensions
                # NOTE: This is a STRUCTURAL MISMATCH - model was trained for embodiment with
                # different action structure than the dataset. For proper evaluation, the dataset
                # should match the model's embodiment action structure.
                # Use action_offset to skip leading dimensions (e.g., base actions for behavior_r1_pro)
                if action_chunk_np.shape[1] > action_dim and action_dim > 0:
                    end_idx = action_offset + action_dim
                    if step_count == 0:
                        # Log which joint groups are being selected based on offset
                        # behavior_r1_pro structure: base(3), torso(4), left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)
                        joint_ranges = {
                            "base": (0, 3),
                            "torso": (3, 7),
                            "left_arm": (7, 14),
                            "left_gripper": (14, 15),
                            "right_arm": (15, 22),
                            "right_gripper": (22, 23),
                        }
                        selected_joints = []
                        for name, (start, end) in joint_ranges.items():
                            if action_offset < end and end_idx > start:
                                overlap_start = max(action_offset, start)
                                overlap_end = min(end_idx, end)
                                selected_joints.append(
                                    f"{name}[{overlap_start - start}:{overlap_end - start}]"
                                )
                        logger.warning(
                            f"Action dimension mismatch: model outputs {action_chunk_np.shape[1]} dims, "
                            f"dataset expects {action_dim} dims. Using dims [{action_offset}:{end_idx}] "
                            f"(action_offset={action_offset}). Selected joint groups: {selected_joints}. "
                            f"For 6-DOF arm dataset, use --action-offset=7 to select left_arm."
                        )
                    action_chunk_np = action_chunk_np[:, action_offset:end_idx]

                # NOTE: predict_action_chunk already calls processor.decode_action() which:
                # 1. Unnormalizes actions using pretrained model's statistics
                # 2. Converts relative->absolute if state is provided
                # Therefore, we should NOT unnormalize again here!

        except Exception as e:
            logger.error(f"Error during inference at step {step_count}: {e}", exc_info=True)
            # Use zeros as fallback
            action_chunk_np = np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Collect predicted actions (aligned with Isaac-GR00T pattern)
        # NOTE: concat_pred_action = action[f"action.{modality_keys[0]}"][j]
        # the np.atleast_1d is to ensure the action is a 1D array
        # Only append inference_interval actions per inference call
        for j in range(min(inference_interval, action_horizon)):
            pred_action_across_time.append(np.atleast_1d(action_chunk_np[j]))

    # Helper function to extract state/action joints (aligned with Isaac-GR00T)
    def extract_state_joints(frames: list[dict], key: str, action_keys_list: list[str] | None = None):
        """Extract and concatenate state/action values across frames."""
        values_list = []
        for frame_data in frames:
            if key in frame_data:
                val = frame_data[key]
                if isinstance(val, torch.Tensor):
                    val = val.cpu().numpy()
                values_list.append(np.atleast_1d(val.flatten()))
            else:
                # Return zeros if key not found
                values_list.append(np.zeros(action_dim, dtype=np.float32))
        return np.vstack(values_list) if values_list else np.array([])

    # Extract ground truth actions (aligned with Isaac-GR00T pattern)
    gt_action_across_time = extract_state_joints(episode_frames[:actual_steps], action_key, action_keys)

    # Convert predictions to numpy array and truncate to actual_steps
    pred_action_across_time = np.array(pred_action_across_time)[:actual_steps]

    # Ensure shapes match (aligned with Isaac-GR00T assertion)
    if gt_action_across_time.shape != pred_action_across_time.shape:
        logger.warning(
            f"Shape mismatch: gt_action {gt_action_across_time.shape}, pred_action {pred_action_across_time.shape}"
        )
        # Pad or truncate to match
        min_dim = min(pred_action_across_time.shape[1], gt_action_across_time.shape[1])
        min_len = min(len(pred_action_across_time), len(gt_action_across_time))
        pred_action_across_time = pred_action_across_time[:min_len, :min_dim]
        gt_action_across_time = gt_action_across_time[:min_len, :min_dim]

    # Compute metrics
    mse = np.mean((gt_action_across_time - pred_action_across_time) ** 2)
    mae = np.mean(np.abs(gt_action_across_time - pred_action_across_time))

    # Print to stdout for visibility (in addition to logging)
    print(f"\n=== Episode {episode_id} Evaluation Results ===")
    print(f"Unnormalized Action MSE: {mse:.6f}")
    print(f"Unnormalized Action MAE: {mae:.6f}")
    print(f"GT action shape: {gt_action_across_time.shape}")
    print(f"Pred action shape: {pred_action_across_time.shape}")
    print("=" * 40)

    logger.info(f"Unnormalized Action MSE across single traj: {mse}")
    logger.info(f"Unnormalized Action MAE across single traj: {mae}")
    logger.info(f"GT action shape: {gt_action_across_time.shape}")
    logger.info(f"Pred action shape: {pred_action_across_time.shape}")

    # Extract state joints for plotting
    state_joints_across_time = []
    for frame in episode_frames[:actual_steps]:
        if state_key in frame:
            state = frame[state_key]
            if isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            if state.ndim == 0:
                state = np.array([state])
            state_joints_across_time.append(state.flatten()[:state_dim])
        else:
            state_joints_across_time.append(np.zeros(state_dim, dtype=np.float32))
    state_joints_across_time = np.array(state_joints_across_time)

    # Plot trajectory results
    if save_plot_path is None:
        # Default to current directory if not specified
        default_dir = Path("./eval_outputs")
        default_dir.mkdir(parents=True, exist_ok=True)
        save_plot_path = str(default_dir / f"traj_{episode_id}.png")

    plot_trajectory_results(
        state_joints_across_time=state_joints_across_time,
        gt_action_across_time=gt_action_across_time,
        pred_action_across_time=pred_action_across_time,
        traj_id=episode_id,
        state_keys=state_keys,
        action_keys=action_keys,
        action_horizon=inference_interval,  # Use inference_interval for marking inference points
        save_plot_path=save_plot_path,
    )

    return mse, mae


@dataclass
class EvalConfig:
    """Configuration for evaluating a GR00T N1.6 policy on a LeRobot v3 dataset.

    This configuration class defines all CLI arguments for the evaluation script.
    Use --help to see all available options.

    Parameter Mapping to Isaac-GR00T (gr00t/eval/open_loop_eval.py):
    ===================================================================
    - embodiment_tag: Maps to ArgsConfig.embodiment_tag (default: NEW_EMBODIMENT)
      * Determines which modality config to use from the processor
      * Available tags: new_embodiment, gr1, behavior_r1_pro, unitree_g1, etc.

    - action_horizon: Maps to ArgsConfig.action_horizon (default: 16)
      * Also determines inference interval (inference every action_horizon steps)
      * SO100 example uses: 16

    - steps: Maps to ArgsConfig.steps (default: 200)
      * SO100 example uses: 400

    - denoising_steps: Maps to ArgsConfig.denoising_steps (default: 4)
      * NOTE: Isaac-GR00T defines this but doesn't use it
      * Model uses config.num_inference_timesteps instead (typically 4)

    - inference_interval: NOT in Isaac-GR00T (always uses action_horizon)
      * LeRobot addition for flexibility, defaults to action_horizon

    Isaac-GR00T Evaluation Pattern:
    ================================
    for step_count in range(0, actual_steps, action_horizon):
        action_chunk = policy.get_action(obs)  # Returns (B, action_horizon, D)
        for j in range(action_horizon):
            pred_action_across_time.append(action_chunk[j])
    """

    dataset_repo_id: str = "izuluaga/finish_sandwich"
    """Hugging Face dataset repository ID (e.g., 'izuluaga/finish_sandwich')."""

    policy_repo_id: str = "nvkartik/gr00t_n1d6-finish_sandwich-relative-action-true-tune-30k"
    """Hugging Face policy repository ID (e.g., 'nvkartik/gr00t_n1d6-finish_sandwich-relative-action-true-tune-30k')."""

    episode_ids: list[int] = field(default_factory=lambda: [0])
    """List of episode IDs to evaluate (e.g., [0] or [0, 1, 2])."""

    steps: int = 200
    """Maximum number of steps to evaluate (will be capped by trajectory length).
    Default matches Isaac-GR00T. SO100 example uses 400 steps."""

    action_horizon: int = 16
    """Action horizon for inference (number of steps predicted at once).
    Also determines the inference interval (inference happens every action_horizon steps)."""

    inference_interval: int | None = None
    """Interval between inference points in steps. If None, uses action_horizon (matches Isaac-GR00T behavior).
    NOTE: Isaac-GR00T always uses action_horizon as the inference interval. This parameter is for advanced use cases only."""

    save_dir: str = "./outputs/eval"
    """Directory to save trajectory plots (e.g., './outputs/eval')."""

    device: str = "cuda"
    """Device to run inference on ('cuda' or 'cpu')."""

    denoising_steps: int | None = None
    """Number of denoising steps for flow matching. If None, uses model's config.num_inference_timesteps.
    NOTE: Isaac-GR00T defines this parameter but doesn't use it - the model uses its config value instead.
    This parameter is included for completeness but may not affect inference."""

    visualize_only: bool = False
    """If True, only visualize dataset trajectories without evaluating the model."""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag for the policy (e.g., 'new_embodiment', 'gr1', 'behavior_r1_pro').
    This determines which modality config to use from the processor.
    Default matches Isaac-GR00T's EmbodimentTag.NEW_EMBODIMENT.
    Available tags: new_embodiment, gr1, behavior_r1_pro, unitree_g1, libero_panda, oxe_google, oxe_widowx."""

    action_offset: int = 0
    """Offset into the model's action output to align with dataset actions.
    For behavior_r1_pro embodiment, action structure is:
      [base(3), torso(4), left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)] = 23 dims
    If your dataset has 6-DOF arm actions, use:
      - action_offset=7 to select left_arm (indices 7-12)
      - action_offset=3 would incorrectly select [torso(4), left_arm_partial(2)]
    Use 0 (default) when model and dataset action structures match."""


def main(config: EvalConfig):
    """Main evaluation function."""
    if config.visualize_only:
        logger.info("Starting dataset trajectory visualization")
    else:
        logger.info("Starting GR00T N1.6 evaluation on LeRobot v3 dataset")
    logger.info(f"Dataset: {config.dataset_repo_id}")
    if not config.visualize_only:
        logger.info(f"Policy: {config.policy_repo_id}")
    logger.info(f"Episodes: {config.episode_ids}")

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset metadata first to check total episodes
    logger.info("Loading dataset metadata...")
    meta = LeRobotDatasetMetadata(config.dataset_repo_id)
    total_episodes = meta.total_episodes
    logger.info(f"Dataset has {total_episodes} total episodes")

    # Validate episode IDs before filtering
    valid_episode_ids = []
    for episode_id in config.episode_ids:
        if episode_id >= total_episodes:
            logger.warning(
                f"Episode ID {episode_id} is out of range (dataset has {total_episodes} episodes). Skipping."
            )
            continue
        valid_episode_ids.append(episode_id)

    if not valid_episode_ids:
        logger.error("No valid episode IDs provided. Exiting.")
        return

    # Now load dataset with filtered episodes
    logger.info(f"Loading dataset with episodes: {valid_episode_ids}")
    dataset = LeRobotDataset(config.dataset_repo_id, episodes=valid_episode_ids)
    logger.info(f"Dataset loaded: {len(dataset)} frames, {dataset.num_episodes} episodes")

    # If visualize_only, skip model loading and evaluation
    if config.visualize_only:
        logger.info("Visualize-only mode: Plotting dataset trajectories without model evaluation")
        for episode_id in valid_episode_ids:
            logger.info(f"Visualizing trajectory for episode: {episode_id}")
            try:
                save_plot_path = str(save_dir / f"traj_{episode_id}.png")
                visualize_dataset_trajectory(
                    dataset,
                    episode_id,
                    steps=config.steps,
                    save_plot_path=save_plot_path,
                )
                logger.info(f"Trajectory plot saved for episode {episode_id}")
            except Exception as e:
                logger.error(f"Error visualizing episode {episode_id}: {e}", exc_info=True)
                continue

        logger.info("Visualization complete!")
        return

    # Load policy for evaluation
    logger.info("Loading policy...")
    policy_class = get_policy_class("gr00t_n1d6")
    policy = policy_class.from_pretrained(config.policy_repo_id)
    policy.eval()
    policy.to(config.device)
    logger.info("Policy loaded successfully")

    # Get embodiment tag from the policy (the policy already determined the correct one during init)
    # The policy's _embodiment_tag was set based on what's available in the processor's modality_configs
    embodiment_tag = policy._embodiment_tag
    available_embodiments = list(policy._processor.modality_configs.keys())

    # Warn if user's requested embodiment tag differs from what the policy is using
    if config.embodiment_tag != embodiment_tag.value:
        logger.warning(
            f"Requested embodiment '{config.embodiment_tag}' differs from policy's embodiment "
            f"'{embodiment_tag.value}'. Using policy's embodiment for correct action decoding. "
            f"Available embodiments: {available_embodiments}"
        )

    # Get modality configs for this embodiment
    modality_configs = policy._processor.modality_configs[embodiment_tag.value]

    logger.info(f"Using embodiment tag: {embodiment_tag.value}")
    logger.info(f"Available embodiments in processor: {available_embodiments}")

    # Log modality keys (aligned with Isaac-GR00T's logging)
    if "action" in modality_configs:
        action_modality_keys = modality_configs["action"].modality_keys
        logger.info(f"Action modality keys: {action_modality_keys}")
        # Log action dimensions and normalization info for debugging
        for key in action_modality_keys:
            norm_params = (
                policy._processor.state_action_processor.norm_params.get(embodiment_tag.value, {})
                .get("action", {})
                .get(key, {})
            )
            if norm_params:
                dim = norm_params.get("dim", "N/A")
                has_min_max = "min" in norm_params and "max" in norm_params
                has_mean_std = "mean" in norm_params and "std" in norm_params
                logger.info(f"  {key}: dim={dim}, has_minmax={has_min_max}, has_meanstd={has_mean_std}")
    if "state" in modality_configs:
        state_modality_keys = modality_configs["state"].modality_keys
        logger.info(f"State modality keys: {state_modality_keys}")

    # Log use_relative_action setting
    use_relative = getattr(policy._processor.state_action_processor, "use_relative_action", "N/A")
    logger.info(f"Processor use_relative_action: {use_relative}")

    # Evaluate each episode
    all_mse = []
    all_mae = []

    for episode_id in valid_episode_ids:
        logger.info(f"Running evaluation on episode: {episode_id}")
        try:
            save_plot_path = str(save_dir / f"traj_{episode_id}.png")
            mse, mae = evaluate_single_trajectory(
                policy,
                dataset,
                episode_id,
                embodiment_tag=embodiment_tag,
                modality_configs=modality_configs,
                steps=config.steps,
                action_horizon=config.action_horizon,
                inference_interval=config.inference_interval,
                save_plot_path=save_plot_path,
                action_offset=config.action_offset,
            )
            logger.info(f"MSE for episode {episode_id}: {mse}, MAE: {mae}")
            all_mse.append(mse)
            all_mae.append(mae)
        except Exception as e:
            logger.error(f"Error evaluating episode {episode_id}: {e}", exc_info=True)
            continue

    # Print summary
    if all_mse:
        avg_mse = np.mean(np.array(all_mse))
        avg_mae = np.mean(np.array(all_mae))
        logger.info(f"Average MSE across all episodes: {avg_mse}")
        logger.info(f"Average MAE across all episodes: {avg_mae}")
    else:
        logger.info("No valid episodes were evaluated.")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(EvalConfig)
    main(config)
