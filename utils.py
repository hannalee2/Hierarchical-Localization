import subprocess
import os
from pathlib import Path
colmap_path = Path(r"C:\Users\hanna.lee\Documents\00_Parallax\000_Project\colmap-x64-windows-cuda\COLMAP.bat")

# Utils
def run_model_orientation_aligner(images_dir: Path, input_path: Path, output_dir: Path):
    print(Path(images_dir))
    os.makedirs(output_dir, exist_ok=True)
    colmap_cmd = [
        str(colmap_path), "model_orientation_aligner",
        "--image_path", str(Path(images_dir)),
        "--input_path", str(input_path),
        "--output_path", str(output_dir), 
        "--method", "MANHATTAN-WORLD"  # {MANHATTAN-WORLD, IMAGE-ORIENTATION}
    ]

    print("[INFO] Running COLMAP model_orientation_aligner...")
    result = subprocess.run(colmap_cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print("[ERROR] COLMAP model_orientation_aligner failed:")
        print(result.stderr)
    else:
        print("[INFO] Model orientation successfully aligned.")
        print("STDOUT:\n", result.stdout)

    return result.returncode == 0  # Return True if successful


def run_model_aligner(input_dir: Path, output_dir: Path, ref_images_path: Path, max_error=0.01):
    os.makedirs(output_dir, exist_ok=True)

    colmap_cmd = [
        str(colmap_path), "model_aligner",
        "--input_path", str(input_dir),
        "--output_path", str(output_dir),
        "--ref_images_path", str(ref_images_path),
        "--ref_is_gps", "0",
        "--alignment_type", "custom", # {plane, custom}
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


def run_model_converter(input_dir: Path, output_dir: Path, output_type="PLY"):
    #output_dir.parent.mkdir(parents=True, exist_ok=True)
    colmap_cmd = [
        str(colmap_path), "model_converter",
        #"--project_path", str(input_dir),
        "--input_path", str(input_dir),
        "--output_path", str("outputs\superpoint+lightglue_4000p\sfm_aligned"),
        "--output_type", output_type, # {BIN, TXT, NVM, Bundler, VRML, PLY, R3D, CAM}        
    ]

    print("[INFO] Running COLMAP model_converter...")
    result = subprocess.run(colmap_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] COLMAP model_converter failed:")
        print(result.stderr)
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