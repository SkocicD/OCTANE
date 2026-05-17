"""Loss computation for OCTANE navigation policy training.

Total loss = Huber(motors)
           + w_bucket  * CrossEntropy(bucket)
           + w_speed   * speed_floor_penalty
           + w_prox    * proximity_penalty      (rocks + craters + walls)
           + w_idle    * idle_penalty
           + w_smooth  * smoothness_penalty     (no jerky consecutive commands)
           + w_diff    * differential_penalty   (no sharp turns during forward travel)
"""

import torch
import torch.nn.functional as F


def compute_loss(motor_pred, bucket_pred, action_gt, terrain, criterion,
                 bucket_loss_weight: float,
                 speed_reg_weight: float,
                 proximity_reg_weight: float,
                 idle_reg_weight: float = 0.003,
                 smooth_reg_weight: float = 0.02,
                 diff_reg_weight: float = 0.01):
    """Compute combined imitation + regularisation loss.

    Args:
        motor_pred:  (B, T, 2)    predicted left/right motor commands
        bucket_pred: (B, T, 3)    predicted bucket state logits
        action_gt:   (B, T, 3)    ground-truth [left, right, bucket_class]
        terrain:     (B, T, 5, H, W)  terrain channels (ch1=rocks, ch2=craters, ch3=walls)
        criterion:   Huber loss instance
        bucket_loss_weight:   weight on CrossEntropy bucket loss
        speed_reg_weight:     weight on speed floor penalty
        proximity_reg_weight: weight on proximity penalty (rocks/craters/walls)
        idle_reg_weight:      tiny weight on idle penalty
        smooth_reg_weight:    weight on temporal smoothness (consecutive step delta)
        diff_reg_weight:      weight on differential stability (no sharp turns while moving forward)

    Returns:
        total_loss (scalar), pred_mag (B, T) per-timestep motor magnitude
    """
    motor_loss  = criterion(motor_pred, action_gt[:, :, :2])
    bucket_loss = F.cross_entropy(
        bucket_pred.reshape(-1, 3),
        action_gt[:, :, 2].reshape(-1).long(),
    )

    pred_mag = motor_pred.abs().mean(dim=-1)   # (B, T)

    # Speed floor: penalise going too slow when the expert is moving.
    moving_mask = (action_gt[:, :, :2].abs().mean(dim=-1) > 0.15).float().detach()
    speed_loss  = (F.relu(0.30 - pred_mag) * moving_mask).mean()

    # Proximity penalty: penalise speed proportional to obstacle/wall presence.
    obs_map  = torch.clamp(terrain[:, :, 1] + terrain[:, :, 2], 0.0, 1.0)
    wall_map = terrain[:, :, 3]
    max_obs  = torch.max(obs_map, wall_map).flatten(2).max(dim=-1).values   # (B, T)
    prox_loss = (max_obs * pred_mag).mean()

    # Idle penalty: tiny pressure against near-zero predictions on all steps.
    idle_loss = F.relu(0.05 - pred_mag).mean()

    # Smoothness: penalise large motor command changes between consecutive steps.
    # Discourages sudden acceleration, deceleration, or direction flips.
    if motor_pred.shape[1] > 1:
        delta       = motor_pred[:, 1:] - motor_pred[:, :-1]   # (B, T-1, 2)
        smooth_loss = delta.pow(2).mean()
    else:
        smooth_loss = motor_pred.new_tensor(0.0)

    # Differential stability: penalise large |left-right| while moving forward.
    # Allows pivoting in place but discourages spinning during forward travel.
    forward_speed = F.relu(motor_pred.mean(dim=-1))             # (B, T), >0 only when moving forward
    diff_penalty  = (motor_pred[:, :, 0] - motor_pred[:, :, 1]).abs()   # (B, T)
    diff_loss     = (diff_penalty * forward_speed).mean()

    total = (motor_loss
             + bucket_loss_weight   * bucket_loss
             + speed_reg_weight     * speed_loss
             + proximity_reg_weight * prox_loss
             + idle_reg_weight      * idle_loss
             + smooth_reg_weight    * smooth_loss
             + diff_reg_weight      * diff_loss)

    return total, pred_mag
