# EKF-SLAM: Extended Kalman Filter SLAM

## Overview

This package implements **Extended Kalman Filter SLAM (EKF-SLAM)** for 2D robot localization using AprilTag landmarks. EKF-SLAM maintains a joint probability distribution over the robot pose and all landmark positions, enabling real-time state estimation with uncertainty quantification.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           EKF-SLAM PIPELINE                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

     ┌──────────────┐         ┌──────────────┐         ┌──────────────┐
     │   Odometry   │         │    Camera    │         │  Camera Info │
     │   /odom      │         │   /camera/   │         │  /camera_    │
     │              │         │  compressed  │         │  node/info   │
     └──────┬───────┘         └──────┬───────┘         └──────┬───────┘
            │                        │                        │
            ▼                        ▼                        │
     ┌──────────────┐         ┌──────────────┐               │
     │   Extract    │         │   Decode     │               │
     │   v, ω, dt   │         │   Image      │               │
     └──────┬───────┘         └──────┬───────┘               │
            │                        │                        │
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
┌───────────────────────────────────────────────────────────────────────────┐
│                         EKF-SLAM BACKEND (ekf_slam.py)                    │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                         STATE VECTOR                                 │  │
│  │    x = [x, y, θ, l₁ₓ, l₁ᵧ, l₂ₓ, l₂ᵧ, ..., lₙₓ, lₙᵧ]ᵀ              │  │
│  │         ├───────┤ ├─────────────────────────────────────┤            │  │
│  │          Robot         Landmark positions                            │  │
│  │          Pose          (2D coordinates)                              │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌────────────────────┐              ┌────────────────────┐               │
│  │   PREDICT STEP     │              │   UPDATE STEP      │               │
│  │   (Odometry)       │              │   (AprilTag)       │               │
│  │                    │              │                    │               │
│  │   • Motion Model   │              │   • Meas. Model    │               │
│  │   • Jacobian Gᵣ    │              │   • Jacobian H     │               │
│  │   • Process Noise Q│              │   • Meas. Noise R  │               │
│  │                    │              │   • Mahal. Gating  │               │
│  │   x̂⁻ = f(x̂, u)    │              │   • Kalman Gain K  │               │
│  │   P⁻ = G·P·Gᵀ + Q  │              │   • State Update   │               │
│  └────────────────────┘              │   • Covariance Upd │               │
│                                      └────────────────────┘               │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    COVARIANCE MATRIX P                              │  │
│  │                                                                     │  │
│  │        ┌─────────┬─────────┬─────────┬─────────┐                   │  │
│  │        │  Pᵣᵣ    │  Pᵣₗ₁   │  Pᵣₗ₂   │  ...    │                   │  │
│  │        │  3×3    │  3×2    │  3×2    │         │                   │  │
│  │        ├─────────┼─────────┼─────────┼─────────┤                   │  │
│  │   P =  │  Pₗ₁ᵣ   │  Pₗ₁ₗ₁  │  Pₗ₁ₗ₂  │  ...    │                   │  │
│  │        │  2×3    │  2×2    │  2×2    │         │                   │  │
│  │        ├─────────┼─────────┼─────────┼─────────┤                   │  │
│  │        │  Pₗ₂ᵣ   │  Pₗ₂ₗ₁  │  Pₗ₂ₗ₂  │  ...    │                   │  │
│  │        │  2×3    │  2×2    │  2×2    │         │                   │  │
│  │        └─────────┴─────────┴─────────┴─────────┘                   │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
            │
            ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │                         OUTPUTS                                  │
     │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │
     │  │ slam_pose  │  │ slam_path  │  │ slam_      │  │ slam_debug │  │
     │  │ PoseStamped│  │ Path       │  │ landmarks  │  │ _image     │  │
     │  │            │  │            │  │ MarkerArray│  │            │  │
     │  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │
     │                                                                  │
     │  ┌────────────┐  ┌────────────┐  ┌────────────┐                  │
     │  │ slam_odom  │  │ slam_pose  │  │ TF: map →  │                  │
     │  │ Odometry   │  │ _cov       │  │ slam_base  │                  │
     │  │            │  │ PoseWith   │  │ _link      │                  │
     │  │            │  │ Covariance │  │            │                  │
     │  └────────────┘  └────────────┘  └────────────┘                  │
     └──────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Foundations

### State Representation

The EKF-SLAM maintains a joint state vector containing the robot pose and all landmark positions:

$$\mathbf{x} = \begin{bmatrix} x \\ y \\ \theta \\ l_{1,x} \\ l_{1,y} \\ \vdots \\ l_{n,x} \\ l_{n,y} \end{bmatrix} \in \mathbb{R}^{3+2n}$$

where $(x, y, \theta)$ is the robot pose and $(l_{i,x}, l_{i,y})$ are landmark positions.

### Covariance Matrix Structure

The covariance matrix captures uncertainty and correlations:

$$\mathbf{P} = \begin{bmatrix} \mathbf{P}_{rr} & \mathbf{P}_{rl_1} & \cdots & \mathbf{P}_{rl_n} \\ \mathbf{P}_{l_1r} & \mathbf{P}_{l_1l_1} & \cdots & \mathbf{P}_{l_1l_n} \\ \vdots & \vdots & \ddots & \vdots \\ \mathbf{P}_{l_nr} & \mathbf{P}_{l_nl_1} & \cdots & \mathbf{P}_{l_nl_n} \end{bmatrix}$$

---

### Prediction Step (Motion Model)

#### Unicycle Motion Model

For a differential-drive robot with linear velocity $v$ and angular velocity $\omega$:

**When $|\omega| \geq \epsilon$ (turning):**

$$\begin{aligned}
x_{k+1} &= x_k + \frac{v}{\omega}\left(\sin(\theta_k + \omega \Delta t) - \sin(\theta_k)\right) \\
y_{k+1} &= y_k + \frac{v}{\omega}\left(-\cos(\theta_k + \omega \Delta t) + \cos(\theta_k)\right) \\
\theta_{k+1} &= \theta_k + \omega \Delta t
\end{aligned}$$

**When $|\omega| < \epsilon$ (straight line):**

$$\begin{aligned}
x_{k+1} &= x_k + v \Delta t \cos(\theta_k) \\
y_{k+1} &= y_k + v \Delta t \sin(\theta_k) \\
\theta_{k+1} &= \theta_k
\end{aligned}$$

#### Motion Jacobian

The Jacobian $\mathbf{G}_{rr}$ with respect to robot state:

**Turning case:**

$$\mathbf{G}_{rr} = \begin{bmatrix} 1 & 0 & \frac{v}{\omega}(\cos(\theta + \omega\Delta t) - \cos\theta) \\ 0 & 1 & \frac{v}{\omega}(\sin(\theta + \omega\Delta t) - \sin\theta) \\ 0 & 0 & 1 \end{bmatrix}$$

**Straight-line case:**

$$\mathbf{G}_{rr} = \begin{bmatrix} 1 & 0 & -v \Delta t \sin\theta \\ 0 & 1 & v \Delta t \cos\theta \\ 0 & 0 & 1 \end{bmatrix}$$

#### Full State Jacobian

The full Jacobian embeds $\mathbf{G}_{rr}$ since landmarks are static:

$$\mathbf{G} = \begin{bmatrix} \mathbf{G}_{rr} & \mathbf{0} \\ \mathbf{0} & \mathbf{I}_{2n} \end{bmatrix}$$

#### Covariance Propagation

$$\mathbf{P}^- = \mathbf{G} \mathbf{P} \mathbf{G}^\top + \mathbf{Q}_d$$

where $\mathbf{Q}_d$ is the discretized process noise affecting only the robot pose:

$$\mathbf{Q}_d = \Delta t \cdot \begin{bmatrix} \mathbf{Q}_{3\times3} & \mathbf{0} \\ \mathbf{0} & \mathbf{0}_{2n\times2n} \end{bmatrix}, \quad \mathbf{Q} = \text{diag}(\sigma_x^2, \sigma_y^2, \sigma_\theta^2)$$

---

### Update Step (Measurement Model)

#### Range-Bearing Measurement Model

Given a landmark at $(l_x, l_y)$ and robot at $(r_x, r_y, r_\theta)$:

$$\mathbf{z} = h(\mathbf{x}) = \begin{bmatrix} r \\ \beta \end{bmatrix} = \begin{bmatrix} \sqrt{(l_x - r_x)^2 + (l_y - r_y)^2} \\ \text{atan2}(l_y - r_y, l_x - r_x) - r_\theta \end{bmatrix}$$

#### Measurement Jacobian

Let $\delta_x = l_x - r_x$, $\delta_y = l_y - r_y$, $q = \delta_x^2 + \delta_y^2$, $\sqrt{q} = r$:

$$\mathbf{H} = \begin{bmatrix} \frac{\partial r}{\partial r_x} & \frac{\partial r}{\partial r_y} & \frac{\partial r}{\partial r_\theta} & \cdots & \frac{\partial r}{\partial l_x} & \frac{\partial r}{\partial l_y} & \cdots \\ \frac{\partial \beta}{\partial r_x} & \frac{\partial \beta}{\partial r_y} & \frac{\partial \beta}{\partial r_\theta} & \cdots & \frac{\partial \beta}{\partial l_x} & \frac{\partial \beta}{\partial l_y} & \cdots \end{bmatrix}$$

**Robot pose partials:**

$$\frac{\partial h}{\partial \mathbf{x}_r} = \begin{bmatrix} -\frac{\delta_x}{\sqrt{q}} & -\frac{\delta_y}{\sqrt{q}} & 0 \\ \frac{\delta_y}{q} & -\frac{\delta_x}{q} & -1 \end{bmatrix}$$

**Landmark position partials:**

$$\frac{\partial h}{\partial \mathbf{l}} = \begin{bmatrix} \frac{\delta_x}{\sqrt{q}} & \frac{\delta_y}{\sqrt{q}} \\ -\frac{\delta_y}{q} & \frac{\delta_x}{q} \end{bmatrix}$$

#### Innovation and Kalman Gain

**Innovation (measurement residual):**

$$\mathbf{y} = \mathbf{z} - h(\mathbf{x}^-)$$

**Innovation covariance:**

$$\mathbf{S} = \mathbf{H} \mathbf{P}^- \mathbf{H}^\top + \mathbf{R}$$

**Kalman gain:**

$$\mathbf{K} = \mathbf{P}^- \mathbf{H}^\top \mathbf{S}^{-1}$$

#### State and Covariance Update

**State update:**

$$\mathbf{x}^+ = \mathbf{x}^- + \mathbf{K} \mathbf{y}$$

**Covariance update (Joseph form for numerical stability):**

$$\mathbf{P}^+ = (\mathbf{I} - \mathbf{K}\mathbf{H}) \mathbf{P}^- (\mathbf{I} - \mathbf{K}\mathbf{H})^\top + \mathbf{K} \mathbf{R} \mathbf{K}^\top$$

---

### Mahalanobis Gating

Before accepting a measurement, we check if it's consistent with predictions:

$$d^2_M = \mathbf{y}^\top \mathbf{S}^{-1} \mathbf{y}$$

**Acceptance criterion:** $d^2_M \leq \chi^2_{\text{threshold}}$

For 2 DOF (range and bearing), typical thresholds:
- 95% confidence: $\chi^2 \approx 5.99$
- 99% confidence: $\chi^2 \approx 9.21$
- Default (loose): $\chi^2 = 25.0$

---

### Landmark Initialization

When a new landmark is first observed, it's added to the state:

#### Position Estimation

$$\begin{bmatrix} l_x \\ l_y \end{bmatrix} = \begin{bmatrix} r_x + r \cos(r_\theta + \beta) \\ r_y + r \sin(r_\theta + \beta) \end{bmatrix}$$

#### Initialization Jacobians

**With respect to robot pose:**

$$\mathbf{G}_r = \begin{bmatrix} 1 & 0 & -r\sin(r_\theta + \beta) \\ 0 & 1 & r\cos(r_\theta + \beta) \end{bmatrix}$$

**With respect to measurement:**

$$\mathbf{G}_z = \begin{bmatrix} \cos(r_\theta + \beta) & -r\sin(r_\theta + \beta) \\ \sin(r_\theta + \beta) & r\cos(r_\theta + \beta) \end{bmatrix}$$

#### New Landmark Covariance

$$\mathbf{P}_{l_\text{new}} = \mathbf{G}_r \mathbf{P}_{rr} \mathbf{G}_r^\top + \mathbf{G}_z \mathbf{R} \mathbf{G}_z^\top + \sigma_\text{init}^2 \mathbf{I}$$

#### Cross-Covariances

**Robot-landmark:**

$$\mathbf{P}_{rl_\text{new}} = \mathbf{P}_{rr} \mathbf{G}_r^\top$$

**Existing landmarks-new landmark:**

$$\mathbf{P}_{l_i l_\text{new}} = \mathbf{P}_{l_i r} \mathbf{G}_r^\top$$

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `Q_xy` | 0.08 m | Process noise std dev for x, y |
| `Q_theta_deg` | 8.0° | Process noise std dev for heading |
| `R_range` | 0.10 m | Measurement noise std dev for range |
| `R_bearing_deg` | 3.0° | Measurement noise std dev for bearing |
| `mahal_gate` | 25.0 | Mahalanobis gating threshold |
| `recovery_inflate_factor` | 1.0 | Covariance inflation on rejection |
| `tag_size` | 0.065 m | AprilTag size |
| `quad_decimate` | 1.0 | Image decimation for detection |

---

## Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Predict | $O(n^2)$ | $O(n^2)$ |
| Update (existing) | $O(n^2)$ | $O(n^2)$ |
| New landmark | $O(n^2)$ | $O(n^2)$ |

where $n = 3 + 2m$ (state dimension with $m$ landmarks).

**Note:** EKF-SLAM has quadratic complexity, making it suitable for small-to-medium maps (~100 landmarks). For larger maps, consider MAP-SLAM with GTSAM.

---

## File Structure

```
slam_package/
├── ekf_slam.py              # EKF-SLAM backend implementation
├── apriltag_ekf_slam_node.py # ROS node frontend
└── README_EKF_SLAM.md       # This documentation
```

---

## Usage

```bash
# Launch the EKF-SLAM node
rosrun slam_package apriltag_ekf_slam_node.py \
    _image_topic:=/camera/compressed \
    _odom_topic:=/odom \
    _tag_size:=0.065

# Visualize in RViz
# Add: MarkerArray → slam_landmarks
# Add: Path → slam_path
# Add: Odometry → slam_odom
```

---

## References

1. Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.
2. Durrant-Whyte, H., & Bailey, T. (2006). Simultaneous localization and mapping: part I. *IEEE Robotics & Automation Magazine*.
3. Bailey, T., & Durrant-Whyte, H. (2006). Simultaneous localization and mapping (SLAM): Part II. *IEEE Robotics & Automation Magazine*.
