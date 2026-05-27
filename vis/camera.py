# General camera visualization utilities
# https://github.com/KwaiVGI/ReCamMaster/blob/main/vis_cam.py
# ttps://github.com/KwaiVGI/ReCamMaster/blob/main/vis_cam.py

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class CameraPoseVisualizer:
    def __init__(self, xlim, ylim, zlim):
        self.fig = plt.figure(figsize=(20, 10))

        # Create two subplots for different views
        self.ax1 = self.fig.add_subplot(121, projection="3d")  # Left subplot
        self.ax2 = self.fig.add_subplot(122, projection="3d")  # Right subplot

        self.plotly_data = None

        # Setup first view (main view)
        self.ax1.set_aspect("auto")
        self.ax1.set_xlim(xlim)
        self.ax1.set_ylim(ylim)
        self.ax1.set_zlim(zlim)
        self.ax1.set_xlabel("x")
        self.ax1.set_ylabel("y")
        self.ax1.set_zlabel("z")
        self.ax1.set_title("Camera Poses - View 1")

        # Setup second view (different perspective)
        self.ax2.set_aspect("auto")
        self.ax2.set_xlim(xlim)
        self.ax2.set_ylim(ylim)
        self.ax2.set_zlim(zlim)
        self.ax2.set_xlabel("x")
        self.ax2.set_ylabel("y")
        self.ax2.set_zlabel("z")
        self.ax2.set_title("Camera Poses - View 2 (Top-down)")

        # Set different viewing angles
        self.ax1.view_init(elev=20, azim=45)  # Default perspective
        self.ax2.view_init(elev=90, azim=0)  # Top-down view

    def extrinsic2pyramid(self, extrinsic, color_map="red", hw_ratio=9 / 16, base_xval=1, zval=3):
        vertex_std = np.array(
            [
                [0, 0, 0, 1],
                [base_xval, -base_xval * hw_ratio, zval, 1],
                [base_xval, base_xval * hw_ratio, zval, 1],
                [-base_xval, base_xval * hw_ratio, zval, 1],
                [-base_xval, -base_xval * hw_ratio, zval, 1],
            ]
        )
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [
            [vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
            [
                vertex_transformed[1, :-1],
                vertex_transformed[2, :-1],
                vertex_transformed[3, :-1],
                vertex_transformed[4, :-1],
            ],
        ]

        # Resolve color: accept named strings, RGBA tuples/arrays, or scalar for rainbow
        if isinstance(color_map, str):
            color = color_map
        elif isinstance(color_map, (tuple, list, np.ndarray)):
            color = color_map
        else:
            color = plt.cm.rainbow(color_map)

        # Add pyramid to both subplots
        self.ax1.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35)
        )
        self.ax2.add_collection3d(
            Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35)
        )

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.viridis(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        # Add legend to the first subplot
        self.ax1.legend(loc="upper left", handles=list_handle)

    def colorbar(self, max_frame_length):
        cmap = mpl.cm.rainbow
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)

        # Create a custom axis for the colorbar at the center bottom of the figure
        cbar_ax = self.fig.add_axes([0.35, 0.02, 0.3, 0.03])  # [left, bottom, width, height]

        cbar = self.fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, orientation="horizontal", label="Frame Number"
        )

    def update_limits(self, xlim, ylim, zlim):
        """Update the limits for both subplots"""
        self.ax1.set_xlim(xlim)
        self.ax1.set_ylim(ylim)
        self.ax1.set_zlim(zlim)

        self.ax2.set_xlim(xlim)
        self.ax2.set_ylim(ylim)
        self.ax2.set_zlim(zlim)

    def show(self):
        plt.suptitle("Camera Poses Visualization - Multiple Views")

        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15, wspace=0.1)

        plt.savefig("tmp/camera_poses.jpg", format="jpg", dpi=300, bbox_inches="tight")
        print(f"Camera poses visualization saved as 'tmp/camera_poses.jpg'")


def visualize_cameras(all_cameras, args):
    """Visualize cameras from all views"""
    visualizer = CameraPoseVisualizer([args.x_min, args.x_max], [args.y_min, args.y_max], [args.z_min, args.z_max])

    # Define colors for different views
    view_colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "gray"]

    all_positions = []
    view_labels = []
    num_views = max(len(all_cameras), 1)
    view_cmap = mpl.colormaps.get_cmap("viridis")

    for view_idx, (view_name, cameras_data) in enumerate(all_cameras.items()):
        cameras = cameras_data["extrinsics"]
        intrinsics = cameras_data["intrinsics"]

        color = view_colors[view_idx % len(view_colors)]
        legend_color = view_cmap(view_idx / num_views)

        # Extract camera positions for automatic scaling
        positions = cameras[:, :3, 3]
        all_positions.extend(positions)

        # Add cameras to visualization
        for frame_idx, c2w in enumerate(cameras):
            # Use frame index for color variation within each view
            frame_color = frame_idx / len(cameras) if len(cameras) > 1 else 0.5

            # Calculate hw_ratio from intrinsics
            cx = intrinsics[frame_idx][0, 2]  # Principal point x
            cy = intrinsics[frame_idx][1, 2]  # Principal point y
            hw_ratio = cy / cx  # height/width ratio

            # Create pyramid for this camera
            # First camera uses the same color as the legend; others vary by frame
            pyramid_color = legend_color if frame_idx == 0 else mpl.colormaps.get_cmap("rainbow")(frame_color)

            visualizer.extrinsic2pyramid(
                c2w,
                color_map=pyramid_color,
                hw_ratio=hw_ratio,
                base_xval=args.base_xval,
                zval=args.zval,
            )

        view_labels.append(view_name)

    # Add legend for views
    if len(view_labels) > 1:
        visualizer.customize_legend(view_labels)

    # Add colorbar for frame progression
    max_frames = max(len(cameras) for cameras in all_cameras.values()) if all_cameras else 1
    visualizer.colorbar(max_frames)

    # Auto-scale the plot based on camera positions
    if all_positions:
        all_positions = np.array(all_positions)
        margin = 0.2
        x_range = [all_positions[:, 0].min() - margin, all_positions[:, 0].max() + margin]
        y_range = [all_positions[:, 1].min() - margin, all_positions[:, 1].max() + margin]
        z_range = [all_positions[:, 2].min() - margin, all_positions[:, 2].max() + margin]

        visualizer.update_limits(x_range, y_range, z_range)

    visualizer.show()
