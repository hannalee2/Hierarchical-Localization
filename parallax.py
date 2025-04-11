import pycolmap
print(f"pycolmap version: {pycolmap.__version__}")
from pathlib import Path
import numpy as np
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
import subprocess
import os
from hloc.utils import viz_3d

from HlocModel import HLocModel

# Setup
images_dir = Path(r"C:\Users\hanna.lee\Documents\00_Parallax\002_TestCode\000_ReticleImages")
#query = "queries/22517664_20250407-144059.png"
query = "queries/Microscope_3_20250403-094514.png"

# Mapping
outputs_dir = Path("outputs/superpoint+lightglue_4000p/")

# Alignment
colmap_path = Path(r"C:\Users\hanna.lee\Documents\00_Parallax\000_Project\colmap-x64-windows-cuda\COLMAP.bat")
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


loc_pairs = outputs_dir / "pairs-loc.txt"

class HLocLocalizer:
    def __init__(self, model, image_dir: Path, output_dir: Path, camera, references, matcher_conf, feature_conf):
        self.model = model
        self.images = Path(image_dir)
        self.outputs = Path(output_dir)
        self.camera = camera
        self.references = references
        self.matcher_conf = matcher_conf
        self.feature_conf = feature_conf

        self.features = self.outputs / "features.h5"
        self.matches = self.outputs / "matches.h5"
        self.loc_pairs = self.outputs / "pairs-loc.txt"

        self.localizer = QueryLocalizer(model, {
            "estimation": {"ransac": {"max_error": 20}},
            "refinement": {"refine_focal_length": False, "refine_extra_params": False}
        })

    def extract_query_features(self, query_path: str):
        print(f"\n[Step 1] Extracting features for {query_path}")
        if self.features.exists():
            print(f"[INFO] features already exist at {self.features}.")
            return
    
        extract_features.main(
            self.feature_conf,
            self.images,
            image_list=[query_path],
            feature_path=self.features,
            overwrite=True
        )

    def match_query_to_references(self, query_path: str):
        print(f"\n[Step 2] Creating and matching pairs for {query_path}")
        if self.loc_pairs.exists() and self.matches.exists():
            print(f"[INFO] Pairs and matches already exist at {self.loc_pairs} and {self.matches}.")
            return
        pairs_from_exhaustive.main(self.loc_pairs, image_list=[query_path], ref_list=self.references)
        match_features.main(
            self.matcher_conf,
            self.loc_pairs,
            features=self.features,
            matches=self.matches,
            overwrite=False
        )

    def estimate_query_pose(self, query_path: str, ref_ids: List[int]):
        print(f"\n[Step 3] Estimating pose for {query_path}")
        ret, log = pose_from_cluster(
            self.localizer, query_path, self.camera, ref_ids, self.features, self.matches
        )
        return ret, log

    def visualize_query_pose(self, query_path: str, ret, log, fig=None):
        print(f"\n[Step 4] Visualizing pose for {query_path}")
        #visualization.visualize_loc_from_log(self.images, query_path, log, self.model)
        
        if fig is None:
            fig = viz_3d.init_figure()
        pose = pycolmap.Image(cam_from_world=ret["cam_from_world"])
        
        viz_3d.plot_camera_colmap(
            fig, pose, self.camera, color="rgba(0,255,0,0.5)", name=query_path, fill=True
        )
        inl_3d = np.array([
            self.model.points3D[pid].xyz for pid in np.array(log["points3D_ids"])[ret["inliers"]]
        ])
        viz_3d.plot_points(fig, inl_3d, color="lime", ps=1, name=query_path)
        fig.show()

    def localize_query(self, query_path: str, ref_ids: List[int], fig):
        print(f"\n[INFO] Localizing query: {query_path}")
        plot_images([read_image(self.images / query_path)], dpi=25)

        self.extract_query_features(query_path)
        self.match_query_to_references(query_path)
        ret, log = self.estimate_query_pose(query_path, ref_ids)
        self.visualize_query_pose(query_path, ret, log, fig)

        print(f"[INFO] Query '{query_path}' localized with {ret['num_inliers']} inliers.")
        return ret, log

# Utils
def run_model_aligner(input_dir: Path, output_dir: Path, ref_images_path: Path, colmap_path: Path, max_error=3.0):
    os.makedirs(output_dir, exist_ok=True)

    colmap_cmd = [
        str(colmap_path), "model_aligner",
        "--input_path", str(input_dir),
        "--output_path", str(output_dir),
        "--ref_images_path", str(ref_images_path),
        "--ref_is_gps", "0",
        "--alignment_type", "plane",
        "--alignment_max_error", str(max_error)
    ]

    print("[INFO] Running COLMAP model_aligner...")
    result = subprocess.run(colmap_cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print("[ERROR] COLMAP model_aligner failed:")
        print(result.stderr)
    else:
        print("[INFO] Model successfully aligned.")
        print("STDOUT:\n", result.stdout)

    return result.returncode == 0  # Return True if successful


def write_points3D_to_ply(model, path):
    with open(path, 'w') as f:
        pts = list(model.points3D.values())
        f.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(len(pts)))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for pt in pts:
            x, y, z = pt.xyz
            r, g, b = pt.color
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

def get_camera():
    return pycolmap.Camera(
        model='OPENCV',
        width=width,
        height=height,
        params=[fx, fy, cx, cy]
    )

mapping_dir = images_dir  / "mapping"
references = [p.relative_to(images_dir).as_posix() for p in mapping_dir.iterdir()]

if __name__ == "__main__":
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

    """
    print("*************** Query ***************")
    # Localize a query image
    hloc_localizer = HLocLocalizer(
        model=model,
        image_dir=images_dir,
        output_dir=outputs_dir,
        camera=get_camera(),
        references=hlocModel.references,
        matcher_conf=hlocModel.matcher_conf,
        feature_conf=hlocModel.feature_conf
    )
    ref_ids = [model.find_image_with_name(r).image_id for r in hlocModel.references]
    print(f"\n[INFO] Found {len(ref_ids)} reference images.")
    hloc_localizer.localize_query(query, ref_ids, hlocModel.fig)

    # Align the model to World space using COLMAP
    success = run_model_aligner(
    input_dir=sfm_dir,
    output_dir=sfm_aligned_dir,
    ref_images_path=camera_centers,
    colmap_path=colmap_path
    )
    

    # Create model from the aligned directory
    aligned_model = pycolmap.Reconstruction(sfm_aligned_dir)
    fig = viz_3d.init_figure()
    all_pts3D = np.array([pt.xyz for pt in aligned_model.points3D.values()])
    viz_3d.plot_points(fig, all_pts3D, color="lime", ps=1, name="all points")
    fig.show()

    hloc_localizer = HLocLocalizer(
        model=aligned_model,
        image_dir=images_dir,
        output_dir=outputs_dir,
        camera=get_camera(),
        references=references,
        matcher_conf=matcher_conf,
        feature_conf=feature_conf
    )

    ref_ids = [aligned_model.find_image_with_name(r).image_id for r in references]
    print(f"\n[INFO] Found {len(ref_ids)} reference images.")
    hloc_localizer.localize_query(query, ref_ids, fig)
    """
    
