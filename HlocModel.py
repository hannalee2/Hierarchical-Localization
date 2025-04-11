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


class HLocModel:
    def __init__(self, image_dir: Path, output_dir: Path, feature_conf, matcher_conf, fx=15400.0, fy=15400.0, cx=2000.0, cy=1500.0, width=4000, height=3000):
        self.image_dir = Path(image_dir)
        self.outputs = Path(output_dir)
        self.mapping_dir = self.image_dir / "mapping"
        self.sfm_pairs = self.outputs / "pairs-sfm.txt"
        self.features = self.outputs / "features.h5"
        self.matches = self.outputs / "matches.h5"
        self.sfm_dir = self.outputs / "sfm"
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.width = width
        self.height = height
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

        self.feature_conf = feature_conf
        self.matcher_conf = matcher_conf
        self.model = None
        self.fig = None

        self.references = [p.relative_to(self.image_dir).as_posix() for p in self.mapping_dir.iterdir()]

        self._setup()

    def _setup(self):
        # Prepare image names (relative to the image folder)
        print(f"[INFO] Found {len(self.references)} mapping images.")

        if self.outputs.exists():
            print(f"[INFO] outputs folder already exist at {self.outputs}.")
            return
        
        # Clear old outputs
        shutil.rmtree(self.outputs, ignore_errors=True)
        self.outputs.mkdir(parents=True, exist_ok=True)

    def extract_features_and_matches(self):
        if not self.features.exists():
            print("\n[INFO] Running feature extraction...")
            extract_features.main(self.feature_conf, self.image_dir, image_list=self.references, feature_path=self.features)
        else:
            print(f"[INFO] Features already exist at {self.features}.")

        if not self.matches.exists():
            print("\n[INFO] Generating exhaustive pairs...")
            pairs_from_exhaustive.main(self.sfm_pairs, image_list=self.references)
            print("\n[INFO] Matching features...")
            match_features.main(self.matcher_conf, self.sfm_pairs, features=self.features, matches=self.matches)
        else:
            print(f"[INFO] Pairs already exist at {self.sfm_pairs}.")

        
    def run_reconstruction(self):
        opts = dict(
            #camera_model='OPENCV',
            camera_model='PINHOLE',
            camera_params=','.join(map(str, (self.fx, self.fy, self.cx, self.cy)))
        )

        mapper_options = dict(
            ba_refine_focal_length=False,
            ba_refine_extra_params=False
        )

        print("\n[INFO] Running reconstruction...")
        self.model = reconstruction.main(
            self.sfm_dir, self.image_dir, self.sfm_pairs,
            self.features, self.matches,
            image_list=self.references,
            image_options=opts,
            mapper_options=mapper_options
        )

        return self.model

    def visualize(self, model):
        if model is None:
            raise RuntimeError("Reconstruction not run yet.")

        print("[INFO] Visualizing in 3D...")
        self.fig = viz_3d.init_figure()
        viz_3d.plot_reconstruction(
            self.fig, model, color="rgba(255,0,0,0.5)",
            name="mapping", points_rgb=True
        )

        #print("[INFO] Showing 2D SfM visualization...")
        #visualization.visualize_sfm_2d(model, self.image_dir, color_by="visibility", n=5)