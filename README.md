# Quackimap: A Duckietown SLAM framework for mapping using AprilTags 

This is the official repository for Quackimap - a project for the Autonomous Vehicles (Duckietown) Course in Fall 2025 at University of Montreal.

**Contributors:**

* Jaydan Aladro (jaydan.aladro@umontreal.ca)
* Arielle Gazzé (arielle.gazze@umontreal.ca)
* Dalil Merad (dalil.merad.1@ens.etsmtl.ca)

# Table of Contents
1. [Running the Code](#running-the-code)
2. [Introduction](#1-introduction)
3. [Related Work](#2-related-work)
4. [Method](#3-method)
   - [AprilTag Detection Pipeline](#31-apriltag-detection-pipeline)
   - [Motion and Measurement Models](#32-motion-and-measurement-models)
   - [EKF-SLAM](#33-ekf-slam)
   - [MAP-SLAM](#34-map-slam)
   - [GTSAM-SLAM](#35-gtsam-slam)
5. [Implementation Details](#4-implementation-details)
6. [Results](#5-results)
7. [Challenges & Lessons Learned](#6-challenges--lessons-learned)
8. [Conclusion](#6-conclusion)
9. [References](#references)


# Running the code
This project has been tested on both a virtual duckiebot as well a physical one. Step 1 and 2 are only relevant for running a virtual bot on the duckiematrix. This project also assumes a proper setup of both the physical bot and the duckiematrix. If this is not your case, please refer to [The Duckietown Manual](https://docs.duckietown.com/ente/duckietown-manual/welcome-to-the-duckietown-manual.html).

### 1. Create virtual bot
If you do not already have a virtual duckiebot, use the following to create one and start sit:
```bash
dts duckiebot virtual create --type duckiebot --configuration DB21J <ROBOT-NAME>
```
```bash
dts duckiebot virtual start <ROBOT-NAME>
```

### 2. Attach and run the Duckiematrix (Simulation Only)

The duckiematrix seems to require absolute paths to the maps, so from the root of this repo, start py assigning a variable

```bash
MAP=$(pwd)/assets/duckiematrix/map
```
Then append the custom map you wish to use within this repo. For now only the loop map is available:
```bash
dts matrix run --standalone --map "$MAP/loop"
```

In another terminal, attach to the matrix:

```bash
dts matrix attach <ROBOT-NAME> map_0/vehicle_0
```

### 3. Build the Package

```bash
dts devel build -f
```
### 4. Run SLAM
Run SLAM launcher against your robot. Use the name of your physical or virtual robot depending on what you wish to use

```bash
dts devel run -R <ROBOT-NAME>
```
By default, the command uses the launcher located at launchers/default.sh but you can specify another one with the -L argument as such:

***TODO a launch file per SLAM implementation instead of just uncommenting on the default***

```bash
dts devel run -R <ROBOT-NAME> -L <your-launcher-name>
```

**Available launchers:**

- `default.sh` — 

### 5. VNC

In another tab/terminal, build the VNC image and run it:
```bash
./launch-vnc.sh <ROBOT-NAME>
```

### 6. Image viewer
If you want to see a live plot of the SLAM:
```bash
dts duckiebot image_viewer <ROBOT-NAME>
```
> NOTE and TODO: only works in EKF SLAM for now

## 1. Introduction

The objective of this project was to learn how to implement a simple SLAM (Simultaneous Localization and Mapping) from scratch and deploy it on the Duckiebot. Both in simulation and on the physical robot.

Duckietown serves as an open-source educational tool for robotics and artificial intelligence. It features a scaled-down city setting where self-driving Duckiebots move through streets and crossroads. These robots come with various sensors including cameras, wheel encoders, and inertial measurement units. A fisheye camera is their main tool for identifying apriltags, AprilTags, which act as fixed landmarks for localization and mapping.

### The SLAM Problem

The main goal of SLAM is constructing and updating a map of an unknown environment while simultaneously keeping track of the robot’s location within it.

* **Mapping**: Given a robot’s location, we need to construct a representation of the world around it.
* **Localization**: Given a map of the robot’s world, we need to determine where the robot is.
* **The Paradox**: Localization assumes a map exists, while mapping assumes localization is solved.

SLAM solves both problems simultaneously by jointly estimating the robot's trajectory and the map of landmarks over time.

### Our Approach

Given the vast body of SLAM literature and the maturity of existing systems, we treated this project as a **learning experience** aimed at understanding the inner workings of different SLAM paradigms. We implemented three complementary approaches:

1. **EKF-SLAM**: a classical filtering approach as our first milestone
2. **MAP-SLAM**: a batch optimization formulation implemented from scratch
3. **GTSAM-SLAM**: a ready available factor graph optimization using the GTSAM library

## 2. Related work
The SLAM problem first emerged towards the end of the 1980's at the IEEE Robotics and Automation Conference [Source] that was being held in San Francisco. Over the next decade, the structure of the problem was clarified and the acronym SLAM was introduced. Since then, it has grown into a major research area with a vast body of litterature and many practical applications in mobile robotics, autonomous driving, augmented/virtual reality, etc. The SLAM problem has been adapted to operate on many sensing modalities such as visual, sonar based or even LiDAR based navigation, making SLAM an interdisciplinary topic at the intersection of many fields. Since the only sensing modality present on the Duckiebots is a mononocular camera, our review focuses on visual SLAM. 

### Evolution of SLAM Approaches

The state-of-the-art has evolved through distinct phases:

| Era | Approach | Key Characteristics |
| --- | --- | --- |
| Early | **EKF-SLAM** | Maintains belief over robot pose and map at every timestep |
| Mid | **Particle Filters** | FastSLAM, Rao-Blackwellized methods for larger environments |
| Modern | **Graph Optimization** | Pose graphs, nonlinear least squares, factor graphs |


### Prior Duckietown SLAM Projects

We reviewed several prior SLAM implementations in the Duckietown ecosystem:

- **SLAMDuck** [4] and **SLAM-Duckietown** [5] : EKF-SLAM implementations using AprilTags and wheel odometry, evaluated with Vicon motion capture, tested using pre-recorded data from rosbags, but not designed for online SLAM.
- **Lane-SLAM** [6] : Builds 2D semantic map of Duckietown lane markings from a previously recorded Duckiebot rosbag using odometry dead-reckoning.

Modern optimization-based visual SLAM can be rougly separated into 2 axes: classical geometry-driven front-ends vs learning augmented front-ends. Most robust SLAM techniques still use pose graph optimization in the backend. Each axis can also be either feature based or pixel/photometric based. The classical feature-based pipilines build sparse correspondences and minimize reprojection error with bundle adjustment and pose-graph optimization. One of the most popular example such a pipeline is ORB-SLAM3 [7]. 


## 3. Method

### 3.1 AprilTag Detection Pipeline
AprilTags are black and white fiducial markers designed to be easy to detect and to encode a unique identifier. They serve as our primary landmarks for SLAM.

#### Detection Pipeline:
1) **Preprocessing**: grayscale conversion + adaptive (tile-based) thresholding to get a clean black/white image.
2) **Segmentation**: group pixels into black/white connected components (union–find).
3) **Quad Detection**: extract boundary clusters and fit quadrilaterals.
4) **Decode**: warp the quad with a homography, sample the bit grid, and decode the tag **ID** (with error checking).

#### Pose Estimation:

1) **Define tag frame**: known 3D corner coordinates in the tag’s local coordinate system.
2) **Extract 2D corners**: from the detector (pixel coordinates of the four corners).
3) **Solve PnP (Perspective-n-Point)**: estimate rotation and translation from 3D–2D correspondences.

More information can be found in the April Tags documentation [10]

We used the [Duckietown AprilTag library](https://github.com/duckietown/lib-dt-apriltags) with the following detector parameters:

| Parameter | Value | Description |
| --- | --- | --- |
| `families` | `tag36h11` | Standard Duckietown tag family |
| `nthreads` | 1-2 | Parallel decoding threads |
| `quad_decimate` | 1.0-2.0 | Image downsampling factor (2.0 for speed) |
| `quad_sigma` | 0.0 | Gaussian blur before detection |
| `refine_edges` | 1   | Sub-pixel edge refinement |
| `decode_sharpening` | 0.25 | Sharpening for decoding |

---

### 3.2 Motion and Measurement Models

All three of our SLAM implementations share the same underlying motion and measurement models.

#### Unicycle Motion Model

The Duckiebot is modeled as a differential driving robot with unicycle kinematics. Given linear velocity $v$ and angular velocity $\omega$ over time interval $\Delta t$:

**Straight-line motion** (when $|\omega| < 10^{-6}$):

$$
\begin{align}
x_{t+1} &= x_t + v \Delta t \cos(\theta_t) \\
y_{t+1} &= y_t + v \Delta t \sin(\theta_t) \\
\theta_{t+1} &= \theta_t
\end{align}
$$

**Arc motion** (when $|\omega| \geq 10^{-6}$):

$$
\begin{align}
x_{t+1} &= x_t + \frac{v}{\omega}\left(\sin(\theta_t + \omega \Delta t) - \sin(\theta_t)\right) \\
y_{t+1} &= y_t + \frac{v}{\omega}\left(-\cos(\theta_t + \omega \Delta t) + \cos(\theta_t)\right) \\
\theta_{t+1} &= \theta_t + \omega \Delta t
\end{align}
$$

The threshold $10^{-6}$ prevents numerical instability when $\omega \approx 0$.

#### Range-Bearing Measurement Model

AprilTag observations are converted to range-bearing measurements in the robot frame:

$$h(x_t, l_j) = \begin{bmatrix} r \\ \beta \end{bmatrix} = \begin{bmatrix} \sqrt{(l_j^x - x_t)^2 + (l_j^y - y_t)^2} \\ \text{atan2}(l_j^y - y_t, l_j^x - x_t) - \theta_t \end{bmatrix}$$

where $(l_j^x, l_j^y)$ is the landmark position and $(x_t, y_t, \theta_t)$ is the robot pose.

#### Camera to Robot Frame Conversion

AprilTag detections provide translation in camera frame $(c_x, c_y, c_z)$ where X is right, Y is down, Z is forward. We convert to robot frame (X forward, Y left):

$$d_x^{\text{robot}} = c_z, \quad d_y^{\text{robot}} = -c_x$$

Then compute range and bearing:
$$r = \sqrt{d_x^2 + d_y^2}, \quad \beta = \text{atan2}(d_y, d_x)$$
---

### 3.3 EKF SLAM

The Extended Kalman Filter formulates SLAM as recursive Bayesian estimation:

$$\underset{x_t, m}{\mathrm{argmax}} \; P(x_t, m \mid z_{1:t}, u_{1:t})$$

where $x_t$ is the robot pose at time $t$, $m$ is the map (landmark positions), $z_{1:t}$ are observations, and $u_{1:t}$ are control inputs.

#### State Representation

The state vector concatenates the robot pose and all landmark positions:

$$
\mathbf{x} = \begin{bmatrix} x & y & \theta & l_1^x & l_1^y & l_2^x & l_2^y & \cdots & l_n^x & l_n^y \end{bmatrix}^T
$$

The covariance matrix $P$ is $(3 + 2n) \times (3 + 2n)$, capturing:

- Robot pose uncertainty (top-left $3 \times 3$ block)
- Landmark uncertainties (diagonal $2 \times 2$ blocks)
- Cross-correlations between robot and landmarks

#### Predict Step:

<!-- TODO we should make our code have the same variable name (F):
$$\begin{align*}
&\bar{\mu}_t = f(\mu_{t-1}, u_t, 0) \\
&\bar{\Sigma}_t = F_t \Sigma_{t-1} F_t^T + W_t Q_t W_t^T
\end{align*}$$
Because right now it's represented as such:
 -->
 
$$ 
\begin{align*}
& \bar{\mu}_t = f(\mu_{t-1}, u_t) \\
& \bar{\Sigma}_t = G_t \Sigma_{t-1} G_t^T + Q_t^{\text{full}}
\end{align*}
$$

where $G_t$ is the full Jacobian (identity except for the robot block $F_t$), and:

$$Q_t^{\text{full}} = \begin{bmatrix} Q \cdot \Delta t & 0 \\ 0 & 0 \end{bmatrix}$$

We scale process noise by $\Delta t$ to account for variable time steps.
 
#### Update Step:

For each landmark observation:

$$
\begin{align*}
&K_t = \bar{\Sigma}_t H_t^T (H_t \bar{\Sigma}_t H_t^T + R)^{-1} \\
&\mu_t = \bar{\mu}_t + K_t(z_t - h(\bar{\mu}_t)) \\
&\Sigma_t = (I - K_t H_t) \bar{\Sigma}_t
\end{align*}
$$

**Residual**: $y = z - h(\bar{\mu}_t)$ with angle wrapping on the bearing component

#### Landmark Initialization

When a new AprilTag is observed for the first time, we initialize it using:

$$\begin{bmatrix} l^x \\ l^y \end{bmatrix} = \begin{bmatrix} x + r\cos(\theta + \beta) \\ y + r\sin(\theta + \beta) \end{bmatrix}$$

The landmark covariance is initialized by propagating uncertainty through this transformation:

$$P_{ll} = G_r P_{rr} G_r^T + G_z R G_z^T + \sigma_{\text{init}}^2 I$$

where:

- $G_r = \frac{\partial l}{\partial (x, y, \theta)}$ — Jacobian w.r.t. robot pose
- $G_z = \frac{\partial l}{\partial (r, \beta)}$ — Jacobian w.r.t. measurement
- $\sigma_{\text{init}}^2 = 0.25$ — Additional initial uncertainty

#### Mahalanobis Gating

To reject outlier measurements, we compute the Mahalanobis distance:

$$d_M^2 = y^T S^{-1} y$$

where $S = H \bar{\Sigma} H^T + R$ is the innovation covariance. Measurements with $d_M^2 > \chi^2_{\text{gate}}$ are rejected.

| Parameter | Value | Description |
| --- | --- | --- |
| `mahal_gate` | 75  | Chi-squared threshold (very permissive for robustness) |

#### Noise Parameters

| Parameter | Simulation | Real Robot | Description |
| --- | --- | --- | --- |
| $\sigma_x, \sigma_y$ | 0.02 m | 0.08 m | Odometry position noise |
| $\sigma_\theta$ | 2°  | 8°  | Odometry heading noise |
| $\sigma_r$ | 0.05 m | 0.10 m | Range measurement noise |
| $\sigma_\beta$ | 3°  | 3.5° | Bearing measurement noise |

The real robot requires higher process noise to account for wheel slip and unmodeled dynamics.

#### Dynamic Measurement Noise

We optionally scale measurement noise with distance to account for increased uncertainty at range:

$$\sigma_r(d) = 0.05 + 0.10 \cdot d$$
$$\sigma_\beta(d) = 3° + d \text{ (in degrees)}$$

---

### 3.4 MAP SLAM

Our custom Maximum A Posteriori (MAP) formulation treats SLAM as a nonlinear least-squares optimization problem over the entire trajectory and map.

#### Problem Formulation

![graph](/images/graph.png)

In maximum-a-posteriori (MAP) SLAM, we are trying to solve the following problem:

$$
\underset{X, L}{\mathrm{argmax}} P(X, L | Z, U) = \underset{X, L}{\mathrm{argmin}} \sum_{i=1}^{M} \underbrace{\| f(x_{i-1}, u_i) - x_i \|_{\Sigma_u}^{-2}}_{\text{odometry factors}} + \sum_{k=1}^{K} \underbrace{\| h(x_{i_k}, l_{j_k}) - z_k \|_{\Sigma_z}^{-2}}_{\text{measurement factors}} + \underbrace{\| x_0 - x_0^{\text{prior}} \|_{\Sigma_0}^{-2}}_{\text{prior factor}}
$$

where the Mahalanobis norm is $\|e\|_{\Sigma}^{-2} = e^T \Sigma^{-1} e$,

and where:

$$
\begin{align*}
&X \text{: M robot poses} \\
&L \text{: K landmarks} \\
&Z \text{: observation of landmarks} \\
&L \text{: controls} \\
&f \text{: motion model} \\
&h \text{: measurement model} \\
\end{align*}
$$

#### State Vector

The full state concatenates all poses and landmarks:

$$\mathbf{s} = \begin{bmatrix} x_0 & y_0 & \theta_0 & x_1 & y_1 & \theta_1 & \cdots & x_M & y_M & \theta_M & l_1^x & l_1^y & \cdots & l_N^x & l_N^y \end{bmatrix}^T$$

Dimension: $3(M+1) + 2N$ where $M$ is number of poses and $N$ is number of landmarks.

#### Gauss-Newton Algorithm

The nonlinear least squares problem is iteratively solved as such:

```
Initialize state s from odometry integration
for iteration = 1 to max_iter:
    Compute residual vector r(s) and Jacobian J
    Solve normal equations: (J^T J + μI) δ = -J^T r
    Update: s ← s + δ
    Wrap all angle states to [-π, π]
    if ||δ|| < ε: break
```

<!-- #### Residual Construction

**Odometry residuals** (3 per pose transition):
$$
r_{\text{odom}}^{(k)} = \begin{bmatrix} 
(x_{k-1} + \Delta x_k) - x_k \\
(y_{k-1} + \Delta y_k) - y_k \\
\text{wrap}((\theta_{k-1} + \Delta\theta_k) - \theta_k)
\end{bmatrix} \cdot \begin{bmatrix} 1/\sigma_x \\ 1/\sigma_y \\ 1/\sigma_\theta \end{bmatrix}
$$

**Measurement residuals** (2 per observation):
$$
r_{\text{meas}}^{(k)} = \begin{bmatrix}
\sqrt{q} - r_{\text{obs}} \\
\text{wrap}(\text{atan2}(\delta_y, \delta_x) - \theta - \beta_{\text{obs}})
\end{bmatrix} \cdot \begin{bmatrix} 1/\sigma_r \\ 1/\sigma_\beta \end{bmatrix}
$$

where $\delta_x = l^x - x$, $\delta_y = l^y - y$, $q = \delta_x^2 + \delta_y^2$.

#### Jacobian Structure

The Jacobian $J$ is sparse with a specific structure:

```
         poses (3M)          landmarks (2N)
       ┌─────────────────┬─────────────────┐
odom   │  block diagonal │       0         │  (3(M-1) rows)
       ├─────────────────┼─────────────────┤
meas   │  sparse entries │  sparse entries │  (2K rows)
       └─────────────────┴─────────────────┘
```

For odometry factor connecting poses $k-1$ and $k$:

- $J[\text{row}, 3(k-1):3k] = \text{diag}(w_x, w_y, w_\theta)$
- $J[\text{row}, 3k:3(k+1)] = \text{diag}(-w_x, -w_y, -w_\theta)$

For measurement factor from pose $i$ to landmark $j$:

- Range row: derivatives of $\sqrt{q}$ w.r.t. pose and landmark
- Bearing row: derivatives of $\text{atan2}$ w.r.t. pose and landmark

#### State Initialization

Good initialization is critical for convergence:

1. **Poses**: Chain odometry forward from initial pose
  $$x_k = x_{k-1} + \Delta x_k, \quad y_k = y_{k-1} + \Delta y_k, \quad \theta_k = \theta_{k-1} + \Delta\theta_k$$
  
2. **Landmarks**: Average of all back-projected observations
  $$l_j = \frac{1}{|O_j|} \sum_{(i,z) \in O_j} \begin{bmatrix} x_i + r \cos(\theta_i + \beta) \\ y_i + r \sin(\theta_i + \beta) \end{bmatrix}$$
  
3. **Warm starting**: When new poses/landmarks are added, we copy the previous solution and only initialize new variables. -->
  

#### Prior on First Pose

To anchor the map frame, we add a strong prior on the first pose:

$$J^T J \leftarrow J^T J + \lambda_{\text{prior}} \cdot \begin{bmatrix} I_{3\times3} & 0 \\ 0 & 0 \end{bmatrix}$$
$$J^T r \leftarrow J^T r + \lambda_{\text{prior}} \cdot (s_{0:3} - s_{0:3}^{\text{prior}})$$

| Parameter | Value | Description |
| --- | --- | --- |
| $\lambda_{\text{prior}}$ | $10^4$ | Prior strength (very strong anchor) |
| $\mu$ | $10^{-6}$ | LM damping factor |
| $\epsilon$ | $10^{-4}$ | Convergence threshold on $\\|\delta\\|$ |
| `max_iter` | 50  | Maximum Gauss-Newton iterations |

#### Noise Parameters

| Parameter | Value | Description |
| --- | --- | --- |
| $\sigma_x, \sigma_y$ | 0.1 m | Odometry position noise |
| $\sigma_\theta$ | 6°  | Odometry heading noise |
| $\sigma_r$ | 0.01 m | Range measurement noise |
| $\sigma_\beta$ | 0.5° | Bearing measurement noise |

Note: The measurement noise is set tighter than EKF because batch optimization can better handle the global consistency.

#### Batch Optimization

---

### 3.5 GTSAM-SLAM

For real time performance, we integrated [GTSAM](https://github.com/borglab/gtsam) (Georgia Tech Smoothing and Mapping).

Factor Graph Formulation

GTSAM represents SLAM as a factor graph where:

- **Variable nodes**: Robot poses $x_i \in SE(2)$ and landmarks $l_j \in \mathbb{R}^2$
- **Factor nodes**: Constraints between variables

```
    x0 ----[odom]---- x1 ----[odom]---- x2 ----[odom]---- x3
    |                  |                  |                 |
 [prior]           [meas]             [meas]            [meas]
                      |                  |                 |
                     l1                 l2                l1
```

#### Factor Types

**Prior Factor** on first pose:
$$f_{\text{prior}}(x_0) \propto \exp\left(-\frac{1}{2}\|x_0 - \mu_0\|_{\Sigma_0}^{-2}\right)$$

**Between Factor** for odometry:
$$f_{\text{odom}}(x_{i-1}, x_i) \propto \exp\left(-\frac{1}{2}\|x_{i-1}^{-1} \oplus x_i - \Delta x\|_{\Sigma_u}^{-2}\right)$$

**Bearing-Range Factor** for measurements:
$$f_{\text{meas}}(x_i, l_j) \propto \exp\left(-\frac{1}{2}\|h(x_i, l_j) - z\|_{\Sigma_z}^{-2}\right)$$

#### iSAM2: Incremental Smoothing

Instead of re-optimizing the entire graph, iSAM2 maintains a Bayes tree and only updates affected variables:

```python
# GTSAM iSAM2 parameters
params = gtsam.ISAM2Params()
params.setRelinearizeThreshold(0.1)  # When to re-linearize variables
params.relinearizeSkip = 1           # Check every update
```

| Parameter | Value | Description |
| --- | --- | --- |
| `relinearizeThreshold` | 0.1 | Relinearize when delta > threshold |
| `relinearizeSkip` | 1   | Check relinearization every N updates |

#### Robust Noise Models

To handle outlier measurements, we wrap Gaussian noise in a Huber robust kernel:

$$\rho_{\text{Huber}}(r) = \begin{cases} \frac{1}{2}r^2 & |r| \leq k \\ k(|r| - \frac{k}{2}) & |r| > k \end{cases}$$

```python
def robust_noise(sigmas, huber_k=1.5):
    base = gtsam.noiseModel.Diagonal.Sigmas(sigmas)
    huber = gtsam.noiseModel.mEstimator.Huber(huber_k)
    return gtsam.noiseModel.Robust.Create(huber, base)
```

| Parameter | Value | Description |
| --- | --- | --- |
| $k$ (Huber) | 1.5 | Transition point to linear loss |


#### Noise Parameters

| Parameter | Value | Description |
| --- | --- | --- |
| $\sigma_x, \sigma_y$ | 0.15 m | Odometry position noise |
| $\sigma_\theta$ | 12° | Odometry heading noise |
| $\sigma_r$ | 0.20 m | Range measurement noise |
| $\sigma_\beta$ | 6°  | Bearing measurement noise |
| $\sigma_{\text{prior}}$ | 0.001 | Prior noise (very tight) |

These are set more conservatively (higher) than the other methods because:

1. Robust noise handles outliers
2. iSAM2 can refine estimates over time
3. Better to be uncertain than overconfident

## 4. Implementation Details

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ROS Node Graph                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐     ┌─────────────────┐                   │
│  │ Camera Node  │────▶│ SLAM Node       │                   │
│  │ (compressed) │     │ (EKF/MAP/GTSAM) │                   │
│  └──────────────┘     └────────┬────────┘                   │
│                                │                            │
│  ┌──────────────┐              │                            │
│  │ Encoder Pose │──────────────┤                            │
│  │    Node      │              │                            │
│  └──────────────┘              ▼                            │
│                       ┌─────────────────┐                   │
│                       │   Publishers    │                   │
│                       │ • slam_pose     │                   │
│                       │ • slam_path     │                   │
│                       │ • slam_landmarks│                   │
│                       │ • slam_odom     │                   │
│                       │ • TF broadcast  │                   │
│                       └─────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### Odometry Source

We used the `lx-kinematics-odometry` [8] wheel encoder odometry processed through `encoder_pose_node`:

- Computes differential wheel rotations from encoder ticks: $\Delta\phi = \frac{\text{ticks} - \text{prev\_ticks}}{\text{resolution}} \cdot 2\pi$
- Applies unicycle kinematics with wheel radius $R = 0.0318$ m and baseline $L = 0.11$ m
- Publishes `nav_msgs/Odometry` with twist information for SLAM prediction

### Camera Intrinsics

Camera parameters are obtained from the `/camera_node/camera_info` topic:

| Parameter | Value |
| --- | --- |
| $f_x$ | ~295.8 |
| $f_y$ | ~299.5 |
| $c_x$ | ~321.3 |
| $c_y$ | ~241.7 |

### Coordinate Frame Conventions

| Frame | Convention |
| --- | --- |
| Camera | X-right, Y-down, Z-forward |
| Robot | X-forward, Y-left |
| World/Map | Standard ROS (X-forward, Y-left) |

### Angle Wrapping

All angle operations use consistent wrapping to $[-\pi, \pi]$:

```python
def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
```


## 5. Results

### EKF-SLAM

The EKF-SLAM implementation served as our foundational milestone. It successfully:

- Tracked the robot pose in real-time
- Initialized and maintained landmark positions
- Provided uncertainty estimates (covariance ellipses)

#### Duckiematrix

Video and images here

#### Real Duckiebot

Video and images here

### MAP-SLAM (From Scratch)

Our custom Gauss Newton optimizer demonstrated the batch optimization concept but revealed practical limitations for online use.

#### Duckiematrix

Video and images here

#### Real Duckiebot

Video and images here

### GTSAM-SLAM

The GTSAM implementation achieved our goal of on real robot performance with robust estimation:

- iSAM2 updates completed within the camera frame period
- Robust noise models handled occasional outlier detections
- Clean trajectory and landmark estimates suitable for navigation

#### Duckiematrix

Video and images here

#### Real Duckiebot

Video and images here


## 6. Challenges & Lessons Learned

### Challenge 1: No Ground Truth for Quantitative Evaluation

The ground truth pose of the physical robot was not available for comparison. We resorted to:

- Visual comparison against known road layouts
- Qualitative assessment of landmark consistency

### Challenge 2: MAP-SLAM Latency

Our custom MAP-SLAM faced a critical real-time limitation:

> *By the time the optimizer finished, the robot had already moved, rendering the correction outdated.*

Even manually stopping the robot couldn't fully resolve this issue due to:

- Poor initialization causing slow convergence
- Linearization errors accumulating between optimizations
- Growing state dimension with more poses/landmarks (complexity $O(n^3)$ for dense solve)

### Challenge 3: Noise Tuning

Finding the right balance of process and measurement noise required extensive experimentation:

| Issue | Symptom | Solution |
| --- | --- | --- |
| Odometry noise too high | Trajectory wanders, jerky corrections | Reduce $Q$ |
| Odometry noise too low | Ignores landmark observations | Increase $Q$ |
| Measurement noise too high | Landmarks not well localized | Reduce $R$ |
| Measurement noise too low | Overconfident, rejects valid measurements | Increase $R$ |

Different parameters were needed for simulation vs. real robot due to:

- Wheel slip on physical surfaces
- Camera calibration differences

### Challenge 4: Landmark Initialization

In EKF-SLAM, initializing landmarks with appropriate uncertainty was crucial:

- Too high initial variance → huge covariance ellipses, slow convergence
- Too low initial variance → overconfident, hard to correct errors

We settled on $\sigma_{\text{init}}^2 = 0.25$ m² as a reasonable compromise.


## 6. Conclusion
This project successfully implemented the EKF SLAM algorithm, a MAP SLAM algorithm and GTSAM library MAP SLAM using apriltags on the virtual Duckiebot in the Duckiematrix and on the Duckibot in real life. Our results show visually that the algorithms work and are seemingly close to the groundtruth. The implementation of three different SLAM algorithms helped us understand the details of the task at hand and the work needed for a SLAM project.


## References

1. [SLAM-Duckietown](https://github.com/AHHHZ975/SLAM-Duckietown)
2. [Duckietown Apriltags library](https://github.com/duckietown/lib-dt-apriltags)
3. [Duckietown](https://duckietown.com)
4. [SLAM slides](https://docs.duckietown.com/daffy/instructor-manual/_assets/slam.pdf)
5. [Graph SLAM slides](https://courses.cs.washington.edu/courses/cse571/23sp/slides/L09/Lecture09_Modern%20SLAM.pdf)
6. [Optimization for SLAM slides](https://www.cs.utexas.edu/~huangqx/2018_CS395_Lecture_13.pdf)
7. [gtsam](https://github.com/borglab/gtsam)
8. [lx-kinematics-odometry](https://github.com/duckietown/lx-kinematics-odometry)
9. [lx-recipe-kinematics-odometry](https://github.com/duckietown/lx-recipe-kinematics-odometry)
10. [lx-recipe-ekf-localization](https://github.com/duckietown/lx-recipe-ekf-localization)

---

<p align="center">
  Built with 🦆 for the Duckietown community
</p>