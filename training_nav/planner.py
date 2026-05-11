"""A* pathfinder on terrain cost maps.

Generates expert trajectories used as training supervision for the
navigation policy. The cost at each cell combines:
  - Terrain slope (gradient of height map)
  - Crater probability
  - Rock probability
  - Wall mask (near-impassable)

Action extraction: looks `lookahead` cells ahead on the A* path from the
robot's current position, computes the desired heading to that waypoint,
then derives (linear_vel, angular_vel) from the angular error to the robot's
current heading.
"""

import heapq
import math

import numpy as np
from scipy.ndimage import sobel


def _build_cost_map(terrain_maps: np.ndarray, cfg: dict) -> np.ndarray:
    """
    terrain_maps: (4, H, W) — height, rocks, craters, walls (robot-centred crop)
    Returns (H, W) float32 cost map.
    """
    pc = cfg['planner']
    height  = terrain_maps[0]
    rocks   = terrain_maps[1]
    craters = terrain_maps[2]
    walls   = terrain_maps[3]

    # Slope magnitude from height
    sx = sobel(height, axis=1)
    sy = sobel(height, axis=0)
    slope = np.sqrt(sx**2 + sy**2)

    cost = (1.0
            + pc['slope_weight']  * slope
            + pc['rock_weight']   * rocks
            + pc['crater_weight'] * craters
            + pc['wall_weight']   * walls)
    return cost.astype(np.float32)


def astar(cost: np.ndarray, start: tuple, goal: tuple) -> list | None:
    """
    A* on a cost grid. start/goal are (row, col) tuples.
    Returns list of (row, col) from start to goal, or None if unreachable.
    """
    rows, cols = cost.shape
    sr, sc = start
    gr, gc = goal

    if not (0 <= sr < rows and 0 <= sc < cols):
        return None
    if not (0 <= gr < rows and 0 <= gc < cols):
        return None

    def h(r, c):
        return math.hypot(r - gr, c - gc)

    open_set = [(h(sr, sc), 0.0, sr, sc)]
    came_from = {}
    g_score = {(sr, sc): 0.0}

    while open_set:
        _, g, r, c = heapq.heappop(open_set)

        if (r, c) == (gr, gc):
            path = []
            node = (r, c)
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append((sr, sc))
            return list(reversed(path))

        if g > g_score.get((r, c), float('inf')) + 1e-6:
            continue

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            step = math.sqrt(2) if (dr and dc) else 1.0
            ng = g + step * float(cost[nr, nc])
            if ng < g_score.get((nr, nc), float('inf')):
                g_score[(nr, nc)] = ng
                came_from[(nr, nc)] = (r, c)
                heapq.heappush(open_set, (ng + h(nr, nc), ng, nr, nc))

    return None


def extract_action(path: list, robot_heading: float,
                   cfg: dict) -> tuple[float, float] | None:
    """
    Given a path (list of (row, col)) and the robot's current heading (radians),
    look `lookahead` steps ahead and compute (linear_vel, angular_vel).

    Convention:
      - heading 0   = +X (right in grid = +col)
      - heading π/2 = +Y (up in grid   = -row, since row 0 is top)

    Returns (linear_vel, angular_vel) both in [-1, 1], or None if path too short.
    """
    lookahead = cfg['planner']['lookahead']
    if len(path) < 2:
        return None

    target_idx = min(lookahead, len(path) - 1)
    tr, tc_ = path[target_idx]
    sr, sc  = path[0]

    # In image coords row increases downward, so negate row difference for heading
    dx = tc_ - sc         # +col = +X
    dy = -(tr - sr)       # +up  = +Y (negate row diff)

    desired_heading = math.atan2(dy, dx)
    angular_error   = _angle_diff(desired_heading, robot_heading)

    # Simple proportional mapping
    max_angular = math.pi / 2   # cap angular vel at 90°/s equivalent
    angular_vel = float(np.clip(angular_error / max_angular, -1.0, 1.0))

    # Reduce linear speed when turning hard
    linear_vel = float(np.clip(1.0 - 0.6 * abs(angular_vel), 0.2, 1.0))

    return linear_vel, angular_vel


def _angle_diff(a: float, b: float) -> float:
    """Signed difference a - b wrapped to [-π, π]."""
    d = a - b
    while d >  math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d


def plan_action(terrain_maps: np.ndarray, goal_heatmap: np.ndarray,
                robot_heading: float, cfg: dict) -> tuple[float, float] | None:
    """
    Full pipeline: build cost map → find goal cell → A* → extract action.
    terrain_maps: (4, gs, gs), goal_heatmap: (gs, gs)
    Returns (linear_vel, angular_vel) or None if no path found.
    """
    gs = cfg['terrain']['grid_size']
    half = gs // 2
    start = (half, half)   # robot always at centre

    cost = _build_cost_map(terrain_maps, cfg)

    # Goal = brightest cell in goal heatmap
    goal_idx = np.unravel_index(np.argmax(goal_heatmap), goal_heatmap.shape)
    goal = (int(goal_idx[0]), int(goal_idx[1]))

    if goal == start or goal_heatmap[goal] < 0.05:
        # Already at goal — stop
        return 0.0, 0.0

    path = astar(cost, start, goal)
    if path is None or len(path) < 2:
        return None

    return extract_action(path, robot_heading, cfg)
