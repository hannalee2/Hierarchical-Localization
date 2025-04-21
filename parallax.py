from pathlib import Path
import numpy as np
import pycolmap
print(f"pycolmap version: {pycolmap.__version__}")
from hloc import (
    extract_features,
    match_features,
    reconstruction,
    visualization,
    pairs_from_exhaustive,
)
from hloc.visualization import plot_images, read_image
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from hloc.utils import viz_3d
import shutil
from typing import List

from hloc.utils import viz_3d

from HlocModel import HLocModel
from HlocLocalizer import HLocLocalizer
from utils import run_model_aligner, run_model_converter, write_points3D_to_ply, run_model_orientation_aligner

# Setup
images_dir = Path(r"C:\Users\hanna.lee\Documents\00_Parallax\002_TestCode\000_ReticleImages")
#query = "queries/22517664_20250407-144059.png"
query = "queries/Microscope_1_20250403-094514.png"

# Mapping
outputs_dir = Path("outputs/superpoint+lightglue_4000p/")

# Alignment
orientation_aligned_dir = outputs_dir / "orientation_aligned"

# Alignment
sfm_aligned_dir = outputs_dir / "sfm_aligned"
features = outputs_dir / "features.h5"
matches = outputs_dir / "matches.h5"
camera_centers = Path("outputs/cam_centers.txt")

# Use LightGlue with DISK features
feature_conf = extract_features.confs["superpoint_parallax"]
matcher_conf = match_features.confs["superpoint+lightglue"]

fx, fy, cx, cy = 15400.0, 15400.0, 2000.0, 1500.0
width, height = 4000, 3000  # or the actual dimensions of your query image
camera = pycolmap.Camera(
    model='OPENCV',
    width=width,
    height=height,
    params=[fx, fy, cx, cy]
)

if __name__ == "__main__":
    """
    print("\n\n*************** Mapping ***************")
    # Model from the database
    hlocModel = HLocModel(
                        image_dir=images_dir,
                        output_dir=outputs_dir,
                        feature_conf=feature_conf,
                        matcher_conf=matcher_conf,
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                        width=width,
                        height=height
                    )
    hlocModel.extract_features_and_matches()
    model = hlocModel.run_reconstruction()
    hlocModel.visualize(model)
    hlocModel.fig.show()

    
    print("*************** Query ***************")
    # Localize a query image
    hloc_localizer = HLocLocalizer(
        model=model,
        image_dir=images_dir,
        output_dir=outputs_dir,
        camera=camera,
        matcher_conf=matcher_conf,
        feature_conf=feature_conf
    )

    hloc_localizer.localize_query(query, hlocModel.fig)
    hlocModel.fig.show()
    """
    
    """
    # Align the model to World space using COLMAP
    print("\n\n*************** Aligning ***************")
    success = run_model_aligner(
                                input_dir=outputs_dir / "sfm",
                                output_dir=sfm_aligned_dir,
                                ref_images_path=camera_centers
                            )
    if success:
        # Create model from the aligned directory
        aligned_model = pycolmap.Reconstruction(sfm_aligned_dir)
        print("[INFO] Visualizing in 3D...")
        fig = viz_3d.init_figure()
        all_pts3D = np.array([pt.xyz for pt in aligned_model.points3D.values()])
        viz_3d.plot_points(fig, all_pts3D, color="lime", ps=1, name="reticle ticks on alinged_model") 
        fig.show()

        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(
            fig, aligned_model, color="rgba(255,0,0,0.5)", name="aligned_model", points_rgb=True
        )
        fig.show()
    
    
    # Align the model to World space using COLMAP
    print("\n\n*************** Manhattan world alignment ***************")
    success = run_model_orientation_aligner(
                                images_dir=images_dir,
                                input_path=sfm_aligned_dir,
                                output_dir=orientation_aligned_dir
                            )
    
    if success:
        # Create model from the aligned directory
        orient_aligned_model = pycolmap.Reconstruction(orientation_aligned_dir)
        print("[INFO] Visualizing in 3D...")
        fig = viz_3d.init_figure()
        all_pts3D = np.array([pt.xyz for pt in orient_aligned_model.points3D.values()])
        viz_3d.plot_points(fig, all_pts3D, color="lime", ps=1, name="reticle ticks on orient_aligned_model") 
        fig.show()

        fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(
            fig, orient_aligned_model, color="rgba(255,0,0,0.5)", name="orient_aligned_model", points_rgb=True
        )
        fig.show()

    #success = run_model_converter(input_dir = sfm_aligned_dir, output_dir=sfm_aligned_dir, output_type="PLY")
    #write_points3D_to_ply(aligned_model, sfm_aligned_dir / "points3D.ply")
    """
 
    print("*************** Query on the Final Model ***************")
    orient_aligned_model = pycolmap.Reconstruction(orientation_aligned_dir)
    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(
        fig, orient_aligned_model, color="rgba(255,0,0,0.5)", name="orient_aligned_model", points_rgb=True
    )
    fig.show()

    hloc_localizer = HLocLocalizer(
        model=orient_aligned_model,
        image_dir=images_dir,
        output_dir=outputs_dir, # store features, matches, paris-loc    
        camera=camera,
        matcher_conf=matcher_conf,
        feature_conf=feature_conf
    )

    hloc_localizer.localize_query(query, fig)
    fig.show()

    # TODO Triangulate the points in the query image using the model
    print("*************** Query on the Final Model ***************")

