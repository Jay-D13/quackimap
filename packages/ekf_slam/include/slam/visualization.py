#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Polygon
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe
import numpy as np
import os
import yaml
import rospkg
import threading
from datetime import datetime
import rospy

class SlamVisualizer:
    """SLAM visualization with ground truth comparison."""

    COLORS = {
        # Background & Grid
        'bg_dark': '#0d1117',
        'bg_panel': '#161b22',
        'grid': '#21262d',
        'grid_minor': '#161b22',
        
        # Text
        'text_primary': '#e6edf3',
        'text_secondary': '#8b949e',
        'text_muted': '#484f58',
        
        # SLAM Elements (Blues/Cyans)
        'slam_path': '#58a6ff',
        'slam_path_gradient_start': '#1f6feb',
        'slam_path_gradient_end': '#58a6ff',
        'slam_robot': '#79c0ff',
        'slam_landmark': '#ff7b72',
        'slam_covariance': '#f0883e',
        
        # Ground Truth Elements (Greens)
        'gt_path': '#3fb950',
        'gt_path_dim': '#238636',
        'gt_landmark': '#56d364',
        'gt_marker': '#2ea043',
        
        # Accent Colors
        'accent_orange': '#f0883e',
        'accent_purple': '#a371f7',
        'accent_pink': '#db61a2',
        'accent_yellow': '#d29922',
        
        # Status Colors
        'success': '#3fb950',
        'warning': '#d29922',
        'error': '#f85149',
    }
    
    def __init__(self, map_name="loop", veh_name="vehicle_0", figsize=(12, 10)):
        self.map_name = map_name
        self.veh_name = veh_name
        self.figsize = figsize
        self._lock = threading.Lock()

        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=figsize, facecolor=self.COLORS['bg_dark'])
        self.ax.set_facecolor(self.COLORS['bg_dark'])

        self.gt_landmarks = {}
        self.gt_path_x, self.gt_path_y = [], []
        self.slam_path_x, self.slam_path_y = [], []
        self.map_tiles = []

        self.frame_count = 0
        self.start_time = None
        self._last_img = None

        self.T_world_to_start = np.eye(3)
        self.tile_size = 0.585

        self._load_ground_truth()
        rospy.loginfo(f"[SlamVisualizer] Initialized: map='{map_name}', vehicle='{veh_name}'")
    
    def _load_ground_truth(self):
        """Load ground truth landmarks and map tiles."""
        try:
            rospack = rospkg.RosPack()
            ekf_slam_path = rospack.get_path('ekf_slam')
            project_root = os.path.abspath(os.path.join(ekf_slam_path, "../.."))
            dynamic_assets_path = os.path.join(project_root, "assets", "duckiematrix", "map", self.map_name)
        except Exception:
            dynamic_assets_path = None

        search_paths = [dynamic_assets_path] if dynamic_assets_path else []
        search_paths.extend([
            f"/code/assets/duckiematrix/map/{self.map_name}",
            f"./assets/duckiematrix/map/{self.map_name}",
        ])

        base_path = next((p for p in search_paths if p and os.path.exists(p)), None)
        if not base_path:
            rospy.logwarn("[SlamVisualizer] Map assets not found.")
            return

        try:
            with open(os.path.join(base_path, "frames.yaml"), 'r') as f:
                frames = yaml.safe_load(f).get('frames', {})

            # Load tile types
            tile_types = {}
            tiles_path = os.path.join(base_path, "tiles.yaml")
            if os.path.exists(tiles_path):
                with open(tiles_path, 'r') as f:
                    for key, val in yaml.safe_load(f).get('tiles', {}).items():
                        tile_types[key] = val.get('type', 'straight')

            # Load tile size
            tile_maps_path = os.path.join(base_path, "tile_maps.yaml")
            if os.path.exists(tile_maps_path):
                with open(tile_maps_path, 'r') as f:
                    tm = yaml.safe_load(f).get('tile_maps', {})
                    if 'map_0' in tm:
                        self.tile_size = tm['map_0'].get('tile_size', {}).get('x', 0.585)

            # Load signs
            signs = {}
            signs_path = os.path.join(base_path, "traffic_signs.yaml")
            if os.path.exists(signs_path):
                with open(signs_path, 'r') as f:
                    signs = yaml.safe_load(f).get('traffic_signs', {})

            # Vehicle transform
            veh_key = f"map_0/{self.veh_name}"
            if veh_key in frames:
                pose = frames[veh_key]['pose']
                x, y, yaw = pose.get('x', 0), -0.18, pose.get('yaw', 0)
                c, s = np.cos(yaw), np.sin(yaw)
                T_start_to_world = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])
                self.T_world_to_start = np.linalg.inv(T_start_to_world)

            start_yaw = np.arctan2(self.T_world_to_start[1, 0], self.T_world_to_start[0, 0])

            # Parse tiles
            for frame_key, frame_data in frames.items():
                if 'tile_' not in frame_key:
                    continue
                pose = frame_data.get('pose', {})
                scale = self.tile_size if frame_data.get('unit', 'meters') == 'tiles' else 1.0
                px, py = pose.get('x', 0) * scale, pose.get('y', 0) * scale
                p_local = self.T_world_to_start @ np.array([px, py, 1.0])
                local_yaw = pose.get('yaw', 0) + start_yaw
                self.map_tiles.append((p_local[0], p_local[1], local_yaw, tile_types.get(frame_key, 'straight')))

            # Parse landmarks
            for frame_key, frame_data in frames.items():
                if 'sign' not in frame_key.lower():
                    continue
                tag_id = signs.get(frame_key, {}).get('id')
                if tag_id is None:
                    for part in reversed(frame_key.split('_')):
                        if part.isdigit():
                            tag_id = int(part)
                            break
                if tag_id is None:
                    continue

                pose = frame_data.get('pose', {})
                scale = self.tile_size if frame_data.get('unit', 'meters') == 'tiles' else 1.0
                px, py = pose.get('x', 0) * scale, pose.get('y', 0) * scale
                p_local = self.T_world_to_start @ np.array([px, py, 1.0])
                self.gt_landmarks[tag_id] = (p_local[0], p_local[1])

            rospy.loginfo(f"[SlamVisualizer] Loaded {len(self.map_tiles)} tiles, {len(self.gt_landmarks)} landmarks")

        except Exception as e:
            rospy.logwarn(f"[SlamVisualizer] Error loading map: {e}")
    
    def update_gt_pose(self, x_global, y_global):
        """Update ground truth path."""
        p_local = self.T_world_to_start @ np.array([x_global, y_global, 1.0])
        self.gt_path_x.append(p_local[0])
        self.gt_path_y.append(p_local[1])
        if len(self.gt_path_x) > 3000:
            self.gt_path_x = self.gt_path_x[-3000:]
            self.gt_path_y = self.gt_path_y[-3000:]

    def update_gt_path(self, x_global, y_global):
        self.update_gt_pose(x_global, y_global)
   
    def render(self, slam_x, slam_P, slam_landmark_ids):
        """
        Render the complete SLAM visualization.
        
        Args:
            slam_x: State vector [x, y, theta, l1_x, l1_y, ...]
            slam_P: Covariance matrix
            slam_landmark_ids: List of landmark tag IDs in order
            
        Returns:
            numpy array: RGB image of the rendered plot
        """
        with self._lock:
            try:
                return self._render_internal(slam_x, slam_P, slam_landmark_ids)
            except Exception as e:
                rospy.logwarn(f"[SlamVisualizer] Render error: {e}")
                if self._last_img is not None:
                    return self._last_img
                h, w = int(self.figsize[1] * 100), int(self.figsize[0] * 100)
                return np.zeros((h, w, 3), dtype=np.uint8)
    
    def _render_internal(self, slam_x, slam_P, slam_landmark_ids):
        if self.start_time is None:
            self.start_time = datetime.now()
        self.frame_count += 1

        # Recreate figure if needed
        try:
            _ = self.ax.get_xlim()
        except Exception:
            try:
                plt.close(self.fig)
            except Exception:
                pass
            self.fig, self.ax = plt.subplots(figsize=self.figsize, facecolor=self.COLORS['bg_dark'])

        self.ax.clear()
        self.ax.set_facecolor(self.COLORS['bg_dark'])

        self._draw_map_tiles()

        rx, ry, rtheta = float(slam_x[0, 0]), float(slam_x[1, 0]), float(slam_x[2, 0])
        self.slam_path_x.append(rx)
        self.slam_path_y.append(ry)
        if len(self.slam_path_x) > 3000:
            self.slam_path_x = self.slam_path_x[-3000:]
            self.slam_path_y = self.slam_path_y[-3000:]

        # GT landmarks
        for tag_id, (gx, gy) in self.gt_landmarks.items():
            self.ax.add_patch(Circle((gx, gy), 0.08, facecolor='none',
                                     edgecolor=self.COLORS['gt_landmark'], linewidth=1.5, alpha=0.3))
            self.ax.add_patch(Circle((gx, gy), 0.04, facecolor=self.COLORS['gt_landmark'],
                                     edgecolor='white', linewidth=1, alpha=0.9))
            self.ax.annotate(f'{tag_id}', (gx, gy), xytext=(8, 8), textcoords='offset points',
                             fontsize=8, fontweight='bold', color=self.COLORS['gt_landmark'],
                             path_effects=[pe.withStroke(linewidth=2, foreground=self.COLORS['bg_dark'])])

        # GT path
        if len(self.gt_path_x) > 1:
            self._draw_gradient_path(self.gt_path_x, self.gt_path_y, self.COLORS['gt_path'], 2, 0.3, 0.8)

        # SLAM path
        if len(self.slam_path_x) > 1:
            self._draw_gradient_path(self.slam_path_x, self.slam_path_y, self.COLORS['slam_path'], 2.5, 0.4, 1.0)

        # SLAM landmarks
        for i, tag_id in enumerate(slam_landmark_ids):
            idx = 3 + 2 * i
            lx, ly = float(slam_x[idx, 0]), float(slam_x[idx + 1, 0])

            if idx + 1 < slam_P.shape[0]:
                self._draw_covariance_ellipse(lx, ly, slam_P[idx:idx + 2, idx:idx + 2])

            self.ax.add_patch(Circle((lx, ly), 0.06, facecolor=self.COLORS['slam_landmark'],
                                     edgecolor='none', alpha=0.2))
            self.ax.add_patch(Circle((lx, ly), 0.035, facecolor=self.COLORS['slam_landmark'],
                                     edgecolor='white', linewidth=1, alpha=0.95))
            self.ax.annotate(f'{tag_id}', (lx, ly), xytext=(8, -18), textcoords='offset points',
                             fontsize=8, fontweight='bold', color=self.COLORS['slam_landmark'],
                             path_effects=[pe.withStroke(linewidth=2, foreground=self.COLORS['bg_dark'])])

            if tag_id in self.gt_landmarks:
                gx, gy = self.gt_landmarks[tag_id]
                self.ax.plot([lx, gx], [ly, gy], color=self.COLORS['accent_yellow'],
                             linestyle=':', linewidth=1, alpha=0.4)

        self._draw_robot(rx, ry, rtheta, slam_P[:3, :3])
        self._draw_stats_overlay(slam_x, slam_landmark_ids)
        self._configure_axes()

        self.fig.tight_layout(pad=0.5)
        self.fig.canvas.draw()

        img_rgb = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()
        self._last_img = img_rgb
        return img_rgb

    def _draw_gradient_path(self, xs, ys, color, linewidth, alpha_start, alpha_end):
        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        n = len(segments)
        colors = np.zeros((n, 4))
        rgb = self._hex_to_rgb(color)
        for i in range(n):
            colors[i] = (*rgb, alpha_start + (alpha_end - alpha_start) * (i / max(n - 1, 1)))
        self.ax.add_collection(LineCollection(segments, colors=colors, linewidths=linewidth))

    def _draw_map_tiles(self):
        if not self.map_tiles:
            return

        half = self.tile_size / 2.0
        base_corners = np.array([[-half, half], [half, half], [half, -half], [-half, -half]])
        lane_kwargs = dict(color='#d29922', linestyle='--', linewidth=1.5, dashes=(4, 4), alpha=0.5, zorder=1)

        for tx, ty, yaw, t_type in self.map_tiles:
            c, s = np.cos(yaw), np.sin(yaw)
            R = np.array([[c, -s], [s, c]])

            def tf(p):
                return R @ np.asarray(p, dtype=float) + np.array([tx, ty])

            corners = (R @ base_corners.T).T + np.array([tx, ty])
            self.ax.add_patch(Polygon(corners, closed=True, facecolor='#21262d',
                                       edgecolor='#30363d', linewidth=1, alpha=0.6, zorder=0))

            if t_type == 'floor':
                continue
            elif t_type == 'straight':
                p1, p2 = tf([0, -half]), tf([0, half])
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], **lane_kwargs)
            elif t_type == 'curve':
                center = np.array([-half, half])
                angles = np.linspace(-np.pi / 2, 0, 48)
                pts = np.stack([center[0] + half * np.cos(angles), center[1] + half * np.sin(angles)], axis=1)
                pts = (R @ pts.T).T + np.array([tx, ty])
                self.ax.plot(pts[:, 0], pts[:, 1], **lane_kwargs)
            elif t_type == '3way':
                stop_kwargs = dict(color='#ff7b72', linewidth=4, alpha=0.8, zorder=2)
                edge_inset = 0.05 * half
                stop_at = half - edge_inset
                for h_local in [np.array([1, 0]), np.array([-1, 0]), np.array([0, -1])]:
                    r_local = np.array([h_local[1], -h_local[0]])
                    p0 = -h_local * stop_at
                    a, b = tf(p0), tf(p0 + r_local * half)
                    self.ax.plot([a[0], b[0]], [a[1], b[1]], **stop_kwargs)

    def _draw_robot(self, x, y, theta, P_robot):
        self.ax.add_patch(Circle((x, y), 0.12, facecolor=self.COLORS['slam_robot'],
                                  edgecolor='none', alpha=0.15))
        self.ax.add_patch(Circle((x, y), 0.07, facecolor=self.COLORS['slam_robot'],
                                  edgecolor='white', linewidth=2, alpha=0.95, zorder=10))

        arrow_len = 0.18
        dx, dy = arrow_len * np.cos(theta), arrow_len * np.sin(theta)
        self.ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                         arrowprops=dict(arrowstyle='-|>', color='white', lw=2.5, mutation_scale=15), zorder=11)

        try:
            self._draw_covariance_ellipse(x, y, P_robot[:2, :2],
                                          color=self.COLORS['slam_robot'], alpha=0.15, linestyle='-', zorder=5)
        except Exception:
            pass

    def _draw_covariance_ellipse(self, x, y, cov, color=None, alpha=0.25, linestyle='--', zorder=3, scale=2.4477):
        if color is None:
            color = self.COLORS['slam_covariance']
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width = min(2 * scale * np.sqrt(max(eigenvalues[0], 1e-10)), 3.0)
            height = min(2 * scale * np.sqrt(max(eigenvalues[1], 1e-10)), 3.0)

            self.ax.add_patch(Ellipse(xy=(x, y), width=width, height=height, angle=angle,
                                       facecolor=color, edgecolor=color, alpha=alpha,
                                       linestyle=linestyle, linewidth=1.5, zorder=zorder))
        except Exception:
            pass

    def _draw_stats_overlay(self, slam_x, slam_landmark_ids):
        rx, ry, rtheta = float(slam_x[0, 0]), float(slam_x[1, 0]), float(slam_x[2, 0])
        n_landmarks = len(slam_landmark_ids)

        error_stats = ""
        if self.gt_path_x and self.slam_path_x:
            dx = self.slam_path_x[-1] - self.gt_path_x[-1]
            dy = self.slam_path_y[-1] - self.gt_path_y[-1]
            error_stats = f"Pos Error: {np.sqrt(dx**2 + dy**2):.3f}m"

        elapsed = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0

        lines = [
            f"╔══ EKF-SLAM STATUS ══╗",
            f"║ Pose: ({rx:+.2f}, {ry:+.2f})",
            f"║ Heading: {np.rad2deg(rtheta):+.1f}°",
            f"║ Landmarks: {n_landmarks}",
            f"║ GT Tags: {len(self.gt_landmarks)}",
        ]
        if error_stats:
            lines.append(f"║ {error_stats}")
        lines.extend([f"║ Time: {elapsed:.1f}s", f"╚═════════════════════╝"])

        props = dict(boxstyle='round,pad=0.5', facecolor=self.COLORS['bg_panel'],
                     edgecolor=self.COLORS['text_muted'], alpha=0.9)
        self.ax.text(0.02, 0.98, "\n".join(lines), transform=self.ax.transAxes,
                     fontsize=9, fontfamily='monospace', verticalalignment='top',
                     color=self.COLORS['text_primary'], bbox=props, zorder=100)

        # legend = "  ".join([f"━━ SLAM Path", f"━━ GT Path", f"● SLAM Tags", f"● GT Tags"])
        # self.ax.text(0.5, 0.02, legend, transform=self.ax.transAxes, fontsize=9,
        #              ha='center', va='bottom', color=self.COLORS['text_secondary'],
        #              path_effects=[pe.withStroke(linewidth=3, foreground=self.COLORS['bg_dark'])], zorder=100)
        handles = [
            Line2D([0], [0], color=self.COLORS['slam_path'], lw=2.5, label='SLAM Path'),
            Line2D([0], [0], color=self.COLORS['gt_path'], lw=2.0, label='GT Path'),
            Line2D([0], [0], marker='o', linestyle='None',
                   markerfacecolor=self.COLORS['slam_landmark'],
                   markeredgecolor='white', markersize=7, label='SLAM Tags'),
            Line2D([0], [0], marker='o', linestyle='None',
                   markerfacecolor=self.COLORS['gt_landmark'],
                   markeredgecolor='white', markersize=7, label='GT Tags'),
        ]

        leg = self.ax.legend(
            handles=handles,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.01),   # inside the axes, near bottom
            ncol=4,
            frameon=False,
            fontsize=9,
            labelcolor=self.COLORS['text_secondary'],
            handlelength=2.0,
            handletextpad=0.6,
            columnspacing=1.2,
            borderaxespad=0.0,
        )


    def _configure_axes(self):
        all_x = self.slam_path_x + self.gt_path_x + [p[0] for p in self.gt_landmarks.values()]
        all_y = self.slam_path_y + self.gt_path_y + [p[1] for p in self.gt_landmarks.values()]

        if all_x and all_y:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            x_range, y_range = x_max - x_min, y_max - y_min
            pad_x, pad_y = max(0.5, x_range * 0.15), max(0.5, y_range * 0.15)

            x_span, y_span = x_range + 2 * pad_x, y_range + 2 * pad_y
            if x_span > y_span:
                pad_y += (x_span - y_span) / 2
            else:
                pad_x += (y_span - x_span) / 2

            self.ax.set_xlim(x_min - pad_x, x_max + pad_x)
            self.ax.set_ylim(y_min - pad_y, y_max + pad_y)

        self.ax.set_aspect('equal')
        self.ax.grid(True, which='major', color=self.COLORS['grid'], linewidth=0.5, alpha=0.5)
        self.ax.grid(True, which='minor', color=self.COLORS['grid_minor'], linewidth=0.3, alpha=0.3)
        self.ax.minorticks_on()
        self.ax.set_xlabel('X (m)', fontsize=10, color=self.COLORS['text_secondary'])
        self.ax.set_ylabel('Y (m)', fontsize=10, color=self.COLORS['text_secondary'])
        self.ax.set_title('EKF-SLAM Live Visualization', fontsize=14, fontweight='bold',
                          color=self.COLORS['text_primary'], pad=10)
        self.ax.tick_params(colors=self.COLORS['text_secondary'], labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color(self.COLORS['grid'])
            spine.set_linewidth(0.5)

    @staticmethod
    def _hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def close(self):
        plt.close(self.fig)