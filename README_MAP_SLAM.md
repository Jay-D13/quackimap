# MAP-SLAM: Maximum A Posteriori SLAM with GTSAM

## Overview

This package implements **MAP-SLAM (Maximum A Posteriori SLAM)** using GTSAM's factor graph framework for 2D robot localization with AprilTag landmarks. Unlike EKF-SLAM which maintains only the current state, MAP-SLAM optimizes over the **entire trajectory**, enabling better handling of loop closures and producing globally consistent maps.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MAP-SLAM PIPELINE (GTSAM)                             │
└─────────────────────────────────────────────────────────────────────────────────┘

     ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
     │   Odometry   │         │    Camera    │         │  Camera Info │
     │   /odom      │         │   /camera/   │         │  /camera_    │
     │              │         │  compressed  │         │  node/info   │
     └──────┬───────┘         └──────┬───────┘         └──────┬───────┘
            │                        │                        │
            ▼                        ▼                        │
     ┌──────────────┐         ┌──────────────┐               │
     │   Compute    │         │   Decode     │               │
     │   Δx, Δy, Δθ │         │   Image      │               │
     │   (local)    │         └──────┬───────┘               │
     └──────┬───────┘                │                        │
            │                        ▼                        ▼
            │                 ┌──────────────┐         ┌──────────────┐
            │                 │  AprilTag    │◄────────│   Camera     │
            │                 │  Detector    │         │  Intrinsics  │
            │                 │  (dt_apriltags)        │  [fx,fy,cx,cy]│
            │                 └──────┬───────┘         └──────────────┘
            │                        │
            │                        ▼
            │                 ┌──────────────┐
            │                 │  Compute     │
            │                 │  Range &     │
            │                 │  Bearing     │
            │                 │  z = [r, β]  │
            │                 └──────┬───────┘
            │                        │
            ▼                        ▼
┌───────────────────────────────────────────────────────────────────────────────┐
│                      MAP-SLAM BACKEND (map_slam_backend.py)                   │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                     FACTOR GRAPH REPRESENTATION                         │  │
│  │                                                                         │  │
│  │   ┌───┐    Odom     ┌───┐    Odom     ┌───┐    Odom     ┌───┐          │  │
│  │   │x₀ │────────────►│x₁ │────────────►│x₂ │────────────►│x₃ │          │  │
│  │   └─┬─┘             └─┬─┘             └─┬─┘             └─┬─┘          │  │
│  │     │                 │                 │                 │            │  │
│  │   Prior             Meas              Meas              Meas           │  │
│  │     │                 │                 │                 │            │  │
│  │     ▼                 ▼                 ▼                 ▼            │  │
│  │   ┌───┐             ┌───┐             ┌───┐             ┌───┐          │  │
│  │   │   │             │l₁ │◄────────────│l₂ │             │l₁ │          │  │
│  │   │   │             └───┘             └───┘             └───┘          │  │
│  │   └───┘                                                                │  │
│  │                                                                         │  │
│  │   Legend:  ○ = Variable (Pose2 or Point2)                              │  │
│  │            ─ = Factor (constraint)                                      │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                         FACTOR TYPES                                    │  │
│  │                                                                         │  │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │  │
│  │  │ PriorFactorPose2│  │BetweenFactorPose2│  │BearingRangeFactor2D│     │  │
│  │  │                 │  │                  │  │                  │         │  │
│  │  │ Anchors x₀ at  │  │ Odometry         │  │ AprilTag         │         │  │
│  │  │ origin          │  │ constraint       │  │ observation      │         │  │
│  │  │ (Pose2)         │  │ xᵢ → xᵢ₊₁       │  │ xᵢ → lⱼ         │         │  │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                       OPTIMIZATION METHODS                              │  │
│  │                                                                         │  │
│  │  ┌──────────────────────────┐    ┌──────────────────────────┐          │  │
│  │  │      iSAM2               │    │   Levenberg-Marquardt    │          │  │
│  │  │   (Incremental)          │    │      (Batch)             │          │  │
│  │  │                          │    │                          │          │  │
│  │  │ • Real-time updates      │    │ • Global optimization    │          │  │
│  │  │ • Sparse Bayes tree      │    │ • Loop closure handling  │          │  │
│  │  │ • O(log n) per update    │    │ • On-demand via service  │          │  │
│  │  │ • Automatic relinear.    │    │ • Full graph solve       │          │  │
│  │  └──────────────────────────┘    └──────────────────────────┘          │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │                        ROBUST NOISE MODEL                               │  │
│  │                                                                         │  │
│  │           Huber Loss                      Standard Least Squares        │  │
│  │               │                                    │                    │  │
│  │        ρ(e)   │    ╱                        ρ(e)   │      ╱            │  │
│  │               │   ╱                               │     ╱              │  │
│  │               │  ╱  Linear for                    │    ╱               │  │
│  │               │ ╱   |e| > k                       │   ╱                │  │
│  │            ───┼╱────────────────              ────┼──╱──────────       │  │
│  │              ╱│    Quadratic                     ╱│                    │  │
│  │             ╱ │    for |e| ≤ k                  ╱ │                    │  │
│  │            ╱  │                                ╱  │                    │  │
│  │           ╱   │                               ╱   │                    │  │
│  │      ────╱────┼────────── e              ────╱────┼─────────── e       │  │
│  │                                                                         │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │                         OUTPUTS                                  │
     │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │
     │  │ slam_pose  │  │ slam_path  │  │ slam_      │  │ slam_debug │  │
     │  │ PoseStamped│  │ Path       │  │ landmarks  │  │ _image     │  │
     │  │ (current)  │  │ (ALL poses)│  │ MarkerArray│  │            │  │
     │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │
     │                                                                  │
     │  ┌────────────┐  ┌────────────┐  ┌────────────┐                  │
     │  │ slam_odom  │  │ slam_pose  │  │ TF: map →  │                  │
     │  │ Odometry   │  │ _cov       │  │ slam_base  │                  │
     │  │            │  │ (optional) │  │ _link      │                  │
     │  └────────────┘  └────────────┘  └────────────┘                  │
     │                                                                  │
     │  ┌────────────────────────────────────────────────────────────┐  │
     │  │              SERVICE: ~optimize                            │  │
     │  │         Triggers batch Levenberg-Marquardt                 │  │
     │  └────────────────────────────────────────────────────────────┘  │
     └──────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Foundations

### Factor Graph Representation

MAP-SLAM represents the SLAM problem as a **factor graph** — a bipartite graph with:
- **Variable nodes** $\mathcal{X} = \{x_0, x_1, \ldots, x_T, l_1, \ldots, l_M\}$ (robot poses and landmarks)
- **Factor nodes** $\mathcal{F} = \{f_0, f_1, \ldots\}$ (constraints from measurements)

The joint probability factors as:

$$p(\mathcal{X} | \mathcal{Z}) \propto \prod_{i} f_i(X_i)$$

where $X_i \subseteq \mathcal{X}$ are variables connected to factor $f_i$.

---

### Maximum A Posteriori Estimation

The MAP estimate maximizes the posterior probability:

$$\mathcal{X}^* = \arg\max_{\mathcal{X}} p(\mathcal{X} | \mathcal{Z})$$

Equivalently, minimize the negative log-likelihood:

$$\mathcal{X}^* = \arg\min_{\mathcal{X}} \sum_i \|h_i(X_i) - z_i\|^2_{\Sigma_i}$$

where $\|e\|^2_{\Sigma} = e^\top \Sigma^{-1} e$ is the Mahalanobis norm.

---

### Factor Types in GTSAM

#### 1. Prior Factor (PriorFactorPose2)

Anchors the first pose at the origin to define the map frame:

$$f_\text{prior}(x_0) = \|x_0 - \bar{x}_0\|^2_{\Sigma_\text{prior}}$$

where $\bar{x}_0 = (0, 0, 0)$ and $\Sigma_\text{prior}$ is very small (high confidence).

#### 2. Odometry Factor (BetweenFactorPose2)

Constrains consecutive poses based on wheel odometry:

$$f_\text{odom}(x_i, x_{i+1}) = \|x_{i+1} \ominus x_i - \Delta_{i,i+1}\|^2_{\Sigma_\text{odom}}$$

where $\ominus$ is the Pose2 "minus" operator (relative transform) and:

$$\Delta_{i,i+1} = \begin{bmatrix} \Delta x \\ \Delta y \\ \Delta \theta \end{bmatrix}$$

is the measured odometry increment in the local frame.

#### 3. Bearing-Range Factor (BearingRangeFactor2D)

Constrains pose-landmark relationships from AprilTag observations:

$$f_\text{tag}(x_i, l_j) = \|h(x_i, l_j) - z_{ij}\|^2_{\Sigma_\text{meas}}$$

where:

$$h(x_i, l_j) = \begin{bmatrix} \text{bearing}(x_i, l_j) \\ \text{range}(x_i, l_j) \end{bmatrix} = \begin{bmatrix} \text{atan2}(l_{j,y} - x_{i,y}, l_{j,x} - x_{i,x}) - \theta_i \\ \sqrt{(l_{j,x} - x_{i,x})^2 + (l_{j,y} - x_{i,y})^2} \end{bmatrix}$$

---

### Robust Noise Models

GTSAM supports robust M-estimators to handle outliers. We use **Huber loss**:

$$\rho_\text{Huber}(e) = \begin{cases} \frac{1}{2}e^2 & \text{if } |e| \leq k \\ k(|e| - \frac{1}{2}k) & \text{if } |e| > k \end{cases}$$

**Properties:**
- Quadratic for small errors (like least squares)
- Linear for large errors (reduces outlier influence)
- Parameter $k$ controls the transition (default: 1.5)

The robust cost function becomes:

$$\sum_i \rho\left(\|h_i(X_i) - z_i\|_{\Sigma_i}\right)$$

---

### Optimization Methods

#### iSAM2 (Incremental Smoothing and Mapping)

GTSAM's iSAM2 maintains a **Bayes tree** — a data structure that enables efficient incremental updates:

1. **Factorization:** The full posterior is factored into a tree structure
2. **Incremental update:** New factors only affect a subset of the tree
3. **Relinearization:** Variables are re-linearized when they move significantly

**Complexity:** $O(\log n)$ per update (amortized), where $n$ is the number of variables.

**Key parameters:**
- `relinearizeThreshold`: When to re-linearize (default: 0.1)
- `relinearizeSkip`: How often to check (default: 1)

#### Levenberg-Marquardt (Batch)

For global optimization (e.g., after loop closure), batch LM solves:

$$\mathcal{X}^{k+1} = \mathcal{X}^k - (J^\top \Sigma^{-1} J + \lambda I)^{-1} J^\top \Sigma^{-1} e$$

where:
- $J$ is the stacked Jacobian of all factors
- $e$ is the stacked error vector
- $\lambda$ is the damping parameter (adjusted adaptively)

**Complexity:** $O(n^3)$ worst case, but exploits sparsity.

---

### Mahalanobis Gating

Before adding a measurement factor, we check consistency:

$$d^2_M = \left(\frac{r - \hat{r}}{\sigma_r}\right)^2 + \left(\frac{\beta - \hat{\beta}}{\sigma_\beta}\right)^2$$

**Acceptance:** $d^2_M \leq \chi^2_\text{threshold}$

This simple diagonal gating is conservative but fast. More sophisticated gating would use the full covariance from marginals.

---

### Landmark Initialization

New landmarks are initialized using the inverse measurement model:

$$\begin{bmatrix} l_x \\ l_y \end{bmatrix} = \begin{bmatrix} x_i + r\cos(\theta_i + \beta) \\ y_i + r\sin(\theta_i + \beta) \end{bmatrix}$$

In GTSAM, this is handled by `pose.transformFrom(Point2(r*cos(β), r*sin(β)))`.

---

### Pose2 Operations in GTSAM

GTSAM's Pose2 class represents SE(2) poses with the following operations:

**Composition (⊕):**
$$x_1 \oplus x_2 = x_1 \cdot x_2 = \begin{bmatrix} R_1 R_2 & R_1 t_2 + t_1 \\ 0 & 1 \end{bmatrix}$$

**Inverse:**
$$x^{-1} = \begin{bmatrix} R^\top & -R^\top t \\ 0 & 1 \end{bmatrix}$$

**Between (⊖):**
$$x_1 \ominus x_2 = x_1^{-1} \cdot x_2$$

**Transform point:**
$$p_\text{world} = R \cdot p_\text{local} + t$$

---

## Comparison: EKF-SLAM vs MAP-SLAM

| Feature | EKF-SLAM | MAP-SLAM (GTSAM) |
|---------|----------|------------------|
| **Representation** | State vector + covariance | Factor graph |
| **What's estimated** | Current pose + all landmarks | Full trajectory + landmarks |
| **Update complexity** | $O(n^2)$ per measurement | $O(\log n)$ incremental |
| **Loop closures** | Requires special handling | Natural through graph |
| **Memory** | $O(n^2)$ (full covariance) | $O(n)$ factors + sparse |
| **Covariance** | Always available | On-demand (expensive) |
| **Past pose updates** | Not possible | Automatic (re-optimization) |
| **Best for** | Small maps, real-time | Large maps, offline/batch |

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Q_xy` | 0.15 m | Odometry noise std dev for x, y |
| `Q_theta_deg` | 12.0° | Odometry noise std dev for heading |
| `R_range` | 0.20 m | Measurement noise std dev for range |
| `R_bearing_deg` | 6.0° | Measurement noise std dev for bearing |
| `huber_k` | 1.5 | Huber robust kernel parameter |
| `gate_chi2` | 25.0 | Mahalanobis gating threshold |
| `enable_gating` | true | Whether to use gating |
| `batch_iters` | 50 | Max iterations for batch optimize |

---

## Services

### `~optimize` (std_srvs/Trigger)

Triggers batch Levenberg-Marquardt optimization over the full graph:

```bash
rosservice call /apriltag_map_slam_node/optimize
```

**Use cases:**
- After detecting a loop closure
- Before saving the map
- When trajectory consistency is critical

---

## File Structure

```
slam_package/
├── map_slam_backend.py        # MAP-SLAM backend with GTSAM
├── apriltag_map_slam_node.py  # ROS node frontend
└── README_MAP_SLAM.md         # This documentation
```

---

## Usage

```bash
# Install GTSAM (if not installed)
pip install gtsam

# Launch the MAP-SLAM node
rosrun slam_package apriltag_map_slam_node.py \
    _image_topic:=/camera/compressed \
    _odom_topic:=/odom \
    _tag_size:=0.065

# Trigger batch optimization
rosservice call /apriltag_map_slam_node/optimize

# Visualize in RViz
# Add: MarkerArray → slam_landmarks
# Add: Path → slam_path (shows FULL optimized trajectory)
# Add: Odometry → slam_odom
```

---

## Key Differences in RViz Visualization

Unlike EKF-SLAM where the path only grows forward, MAP-SLAM's `slam_path` shows the **entire optimized trajectory**. After batch optimization or loop closure, you may see past poses "snap" to more accurate positions — this is the graph optimizer correcting historical errors.

The blue trajectory line in the `slam_landmarks` MarkerArray is rebuilt from all poses after each update, ensuring consistency with the current estimate.

---

## References

1. Dellaert, F., & Kaess, M. (2017). Factor Graphs for Robot Perception. *Foundations and Trends in Robotics*.
2. Kaess, M., Johannsson, H., Roberts, R., Ila, V., Leonard, J. J., & Dellaert, F. (2012). iSAM2: Incremental smoothing and mapping using the Bayes tree. *IJRR*.
3. Grisetti, G., Kümmerle, R., Stachniss, C., & Burgard, W. (2010). A tutorial on graph-based SLAM. *IEEE Intelligent Transportation Systems Magazine*.
4. GTSAM Documentation: https://gtsam.org/
