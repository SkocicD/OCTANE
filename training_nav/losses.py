"""Loss computation for OCTANE navigation policy training.

Total loss = Huber(motors)
           + w_bucket  * CrossEntropy(bucket)
           + w_speed   * speed_floor_penalty
           + w_prox    * proximity_penalty      (rocks + craters + walls)
           + w_idle    * idle_penalty
"""

import torch
import torch.nn.functional as F


def compute_loss(motor_pred, bucket_pred, action_gt, terrain, criterion,
                 bucket_loss_weight: float,
                 speed_reg_weight: float,
                 proximity_reg_weight: float,
                 idle_reg_weight: float = 0.003):
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
        idle_reg_weight:      tiny weight on idle penalty (discourages zero-output predictions)

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
    # Prevents mode collapse toward zero without fighting expert speed.
    moving_mask = (action_gt[:, :, :2].abs().mean(dim=-1) > 0.15).float().detach()
    speed_loss  = (F.relu(0.30 - pred_mag) * moving_mask).mean()

    # Proximity penalty: rocks (ch1) + craters (ch2) + walls (ch3).
    # Walls use ch3 which is 1.0 at arena boundaries and out-of-bounds fill —
    # this correctly penalises the model for commanding speed near any wall.
    # Clamp rocks+craters to [0,1] before max so wall (binary 0/1) isn't swamped.
    obs_map  = torch.clamp(terrain[:, :, 1] + terrain[:, :, 2], 0.0, 1.0)
    wall_map = terrain[:, :, 3]
    max_obs  = torch.max(obs_map, wall_map).flatten(2).max(dim=-1).values   # (B, T)
    prox_loss = (max_obs * pred_mag).mean()

    # Idle penalty: tiny constant pressure against near-zero motor predictions.
    # Fires on ALL steps (not masked by expert) — magnitude is intentionally
    # small so it cannot override correct stationary expert steps (dumping phase).
    idle_loss = F.relu(0.05 - pred_mag).mean()

    total = (motor_loss
             + bucket_loss_weight   * bucket_loss
             + speed_reg_weight     * speed_loss
             + proximity_reg_weight * prox_loss
             + idle_reg_weight      * idle_loss)

    return total, pred_mag
