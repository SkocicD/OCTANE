"""Loss computation for OCTANE navigation policy training.

Total loss = Huber(motors)
           + w_bucket     * CrossEntropy(bucket)
           + w_speed      * speed_floor_penalty
           + w_prox       * proximity_penalty       (rocks + craters + walls)
           + w_idle       * idle_penalty
           + w_smooth     * smoothness_penalty       (no jerky consecutive commands)
           + w_diff       * differential_penalty     (no sharp turns during forward travel)
           + w_forward    * backward_penalty         (nav phases should move forward)
           + w_stillness  * stillness_penalty        (no motion during dumping)
"""

import torch
import torch.nn.functional as F


def compute_loss(motor_pred, bucket_pred, action_gt, terrain, criterion,
                 bucket_loss_weight: float,
                 speed_reg_weight: float,
                 proximity_reg_weight: float,
                 idle_reg_weight: float        = 0.003,
                 smooth_reg_weight: float      = 0.02,
                 diff_reg_weight: float        = 0.01,
                 forward_reg_weight: float     = 0.02,
                 stillness_reg_weight: float   = 0.05,
                 recovery_reg_weight: float    = 0.05):
    """Compute combined imitation + regularisation loss.

    Args:
        motor_pred:  (B, T, 2)       predicted left/right motor commands
        bucket_pred: (B, T, 3)       predicted bucket state logits
        action_gt:   (B, T, 3)       ground-truth [left, right, bucket_class]
        terrain:     (B, T, 5, H, W) terrain channels (ch1=rocks, ch2=craters, ch3=walls)
        criterion:   Huber loss instance
        bucket_loss_weight:    weight on CrossEntropy bucket loss
        speed_reg_weight:      weight on speed floor penalty
        proximity_reg_weight:  weight on proximity penalty
        idle_reg_weight:       weight on idle penalty
        smooth_reg_weight:     weight on temporal smoothness (consecutive step delta)
        diff_reg_weight:       weight on differential stability (no spinning while driving forward)
        forward_reg_weight:    weight on backward motion penalty (nav phases only)
        stillness_reg_weight:  weight on motion during dumping penalty

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

    # Proximity: penalise speed near obstacles/walls, but exempt steps where the
    # expert is already in recovery (backing up).  Penalising backup speed near
    # obstacles would teach the model to stop instead of escaping.
    obs_map      = torch.clamp(terrain[:, :, 1] + terrain[:, :, 2], 0.0, 1.0)
    wall_map     = terrain[:, :, 3]
    max_obs      = torch.max(obs_map, wall_map).flatten(2).max(dim=-1).values  # (B, T)
    not_recovery = (action_gt[:, :, :2].mean(dim=-1) >= -0.05).float().detach()
    prox_loss    = (max_obs * pred_mag * not_recovery).mean()

    # Idle: tiny pressure against near-zero predictions on all steps.
    idle_loss = F.relu(0.05 - pred_mag).mean()

    # Smoothness: penalise large motor delta between consecutive timesteps.
    if motor_pred.shape[1] > 1:
        delta       = motor_pred[:, 1:] - motor_pred[:, :-1]   # (B, T-1, 2)
        smooth_loss = delta.pow(2).mean()
    else:
        smooth_loss = motor_pred.new_tensor(0.0)

    # Differential stability: penalise |left-right| weighted by forward speed.
    # Allows pivoting in place; discourages spinning during forward travel.
    forward_speed = F.relu(motor_pred.mean(dim=-1))                       # (B, T)
    diff_penalty  = (motor_pred[:, :, 0] - motor_pred[:, :, 1]).abs()    # (B, T)
    diff_loss     = (diff_penalty * forward_speed).mean()

    # Forward bias: penalise reverse travel during navigation phases (bucket_gt=0),
    # BUT only when the expert is also going forward.  When the expert backs up
    # (recovery from a wall/obstacle), we must NOT fight that signal — the model
    # needs to learn backup behaviour from those samples.
    nav_mask        = (action_gt[:, :, 2] == 0).float().detach()
    gt_fwd_mask     = (action_gt[:, :, :2].mean(dim=-1) > 0.05).float().detach()
    avg_motor       = motor_pred.mean(dim=-1)                             # (B, T)
    forward_loss    = (F.relu(-avg_motor) * nav_mask * gt_fwd_mask).mean()

    # Stillness: penalise any motor output during dumping (bucket_gt=2).
    # Robot must be completely stopped at the berm while depositing.
    dump_mask      = (action_gt[:, :, 2] == 2).float().detach()
    stillness_loss = (pred_mag * dump_mask).mean()

    # Recovery encouragement: when the expert backs up (GT avg motor < -0.05)
    # near an obstacle, reward the model for also outputting negative motors.
    # This gives a direct gradient signal for escape behaviour.
    expert_backing = (action_gt[:, :, :2].mean(dim=-1) < -0.05).float().detach()
    near_obs       = (max_obs > 0.3).float().detach()
    recovery_loss  = (F.relu(avg_motor) * expert_backing * near_obs).mean()

    total = (motor_loss
             + bucket_loss_weight    * bucket_loss
             + speed_reg_weight      * speed_loss
             + proximity_reg_weight  * prox_loss
             + idle_reg_weight       * idle_loss
             + smooth_reg_weight     * smooth_loss
             + diff_reg_weight       * diff_loss
             + forward_reg_weight    * forward_loss
             + stillness_reg_weight  * stillness_loss
             + recovery_reg_weight   * recovery_loss)

    return total, pred_mag
