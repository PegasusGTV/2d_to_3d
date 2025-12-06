import torch
import numpy as np
import open3d as o3d
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import cv2
import os

# Try to import rembg, make it optional
try:
    from rembg import remove
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("Warning: rembg not installed. Background removal will be disabled.")
    print("Install with: pip install rembg")

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEPTH_CHECKPOINT = "depth-anything/Depth-Anything-V2-Large-hf"
FLYING_PIXEL_THRESHOLD = 0.15  # Sensitivity for edge cleanup (lower = more aggressive)
REMOVE_BACKGROUND = True       # Set to False if you ever want the background back

class DepthEstimator:
    def __init__(self):
        print(f"Loading Depth Anything V2 (Large) on {DEVICE}...")
        self.image_processor = AutoImageProcessor.from_pretrained(DEPTH_CHECKPOINT)
        self.model = AutoModelForDepthEstimation.from_pretrained(DEPTH_CHECKPOINT).to(DEVICE)
        self.model.eval()

    def estimate(self, image_pil):
        # Prepare image
        inputs = self.image_processor(images=image_pil, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image_pil.size[::-1], 
            mode="bicubic",
            align_corners=False,
        )

        depth = prediction.squeeze().cpu().numpy()
        
        # Normalize to 0-1
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max > depth_min:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        else:
            depth_normalized = np.zeros_like(depth)
        
        return depth_normalized

def get_subject_mask(image_pil):
    """
    Uses rembg to create a binary mask of the subject.
    Returns: (H, W) boolean numpy array where True = Keep, False = Remove
    """
    if not REMBG_AVAILABLE:
        print("Warning: rembg not available. Using full image mask.")
        return np.ones((image_pil.size[1], image_pil.size[0]), dtype=bool)
    
    print("Running AI background removal...")
    
    # rembg expects and returns PIL images
    # This returns an RGBA image where the background is transparent (A=0)
    result = remove(image_pil)
    
    # Extract alpha channel
    if result.mode == 'RGBA':
        alpha = np.array(result)[:, :, 3]
    else:
        # Fallback if something weird happens, though rembg usually returns RGBA
        print("Warning: Background removal did not return RGBA. Using heuristic.")
        alpha = np.ones((result.size[1], result.size[0])) * 255

    # Create binary mask (threshold at 128 to be safe)
    mask = alpha > 10

    # Optional: Erode slightly to remove the "white halo" often found on cutouts
    mask_uint8 = mask.astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    mask_eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    
    return mask_eroded.astype(bool)

def filter_flying_pixels(depth_map, threshold=0.05):
    """
    Removes pixels on the "slopes" of steep depth changes.
    """
    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    edge_mask = magnitude < threshold
    return edge_mask

def depth_to_pointcloud(pil_img, depth, mask=None, depth_scale=10.0):
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    
    # If image has alpha channel, ignore it for the color mapping, use just RGB
    if img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]

    H, W = depth.shape
    fx = max(H, W) * 1.2
    fy = max(H, W) * 1.2
    cx = W / 2.0
    cy = H / 2.0

    # Invert depth: 0=far, 1=close -> Z distance
    Z = depth_scale / (depth + 0.1) 

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy

    # Flatten everything
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z.reshape(-1)
    colors = img_np.reshape(-1, 3)
    
    # Start with all true
    valid_mask = np.ones_like(Z, dtype=bool)
    
    # Apply user mask (Background removal + Flying pixels)
    if mask is not None:
        valid_mask = valid_mask & mask.reshape(-1)

    # Filter invalid math (inf/nan)
    valid_mask = valid_mask & np.isfinite(Z) & (Z > 0)
    
    # Apply filter
    points = np.stack([X[valid_mask], -Y[valid_mask], -Z[valid_mask]], axis=1)
    colors = colors[valid_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def cleanup_pointcloud(pcd):
    print("Running statistical outlier removal...")
    # These settings are tuned for "clean characters"
    # nb_neighbors: 50 -> looks at 50 nearest points
    # std_ratio: 1.5 -> removes points that are > 1.5 deviations away (dust)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.5)
    cleaned_pcd = pcd.select_by_index(ind)
    print(f"Removed {len(pcd.points) - len(cleaned_pcd.points)} noise points")
    return cleaned_pcd

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. LOAD IMAGE
    # ---------------------------------------------------------
    img_path = os.path.join(script_dir, "results/test_images/pikachu_inpainted.png")
    
    # Try alternative path if not found
    if not os.path.exists(img_path):
        img_path = "results/test_images/pikachu_inpainted.png"
    
    try:
        # We process original RGB for depth, but keep it for masking
        img_original = Image.open(img_path).convert("RGB")
        print(f"Loaded: {img_path}")
        print(f"Image size: {img_original.size}")
    except FileNotFoundError:
        print(f"Error: Could not find image at {img_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Script directory: {script_dir}")
        exit(1)

    # 2. GENERATE MASKS (The most important part for you)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("Step 1: Generating masks...")
    print("="*60)
    
    # A. Background Mask (Subject Isolation)
    if REMOVE_BACKGROUND:
        if not REMBG_AVAILABLE:
            print("Note: rembg not installed. Background removal disabled.")
            print("      Install with: pip install rembg")
            print("      Continuing without background removal...")
        bg_mask = get_subject_mask(img_original)
        print(f"Background mask: {bg_mask.sum()} / {bg_mask.size} pixels kept ({100*bg_mask.sum()/bg_mask.size:.1f}%)")
    else:
        bg_mask = np.ones((img_original.size[1], img_original.size[0]), dtype=bool)
        print("Background removal disabled (REMOVE_BACKGROUND=False)")
        
    # B. Depth Estimation
    print("\n" + "="*60)
    print("Step 2: Estimating depth...")
    print("="*60)
    estimator = DepthEstimator()
    depth = estimator.estimate(img_original)
    print(f"Depth map shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")
    
    # C. Flying Pixel Mask (Geometric cleanup)
    print("\n" + "="*60)
    print("Step 3: Filtering flying pixels...")
    print("="*60)
    geometry_mask = filter_flying_pixels(depth, threshold=FLYING_PIXEL_THRESHOLD)
    print(f"Geometry mask: {geometry_mask.sum()} / {geometry_mask.size} pixels kept ({100*geometry_mask.sum()/geometry_mask.size:.1f}%)")
    
    # Combine masks: Must be Subject AND Geometry valid
    final_mask = bg_mask & geometry_mask
    print(f"Final combined mask: {final_mask.sum()} / {final_mask.size} pixels kept ({100*final_mask.sum()/final_mask.size:.1f}%)")

    # 3. GENERATE & SAVE
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("Step 4: Generating point cloud...")
    print("="*60)
    pcd = depth_to_pointcloud(img_original, depth, mask=final_mask)
    print(f"Initial point cloud: {len(pcd.points)} points")
    
    # Statistical cleanup to remove floating "dust" around the cut
    print("\n" + "="*60)
    print("Step 5: Cleaning point cloud...")
    print("="*60)
    pcd_clean = cleanup_pointcloud(pcd)
    print(f"Final point cloud: {len(pcd_clean.points)} points")

    print("\n" + "="*60)
    print("Step 6: Saving results...")
    print("="*60)
    output_path = os.path.join(script_dir, "pikachu_isolated.ply")
    o3d.io.write_point_cloud(output_path, pcd_clean)
    print(f"SUCCESS! Saved to: {output_path}")

    # 4. VISUALIZE
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print("Step 7: Visualization...")
    print("="*60)
    try:
        if os.getenv('DISPLAY') is not None:
            print("Opening visualization window...")
            o3d.visualization.draw_geometries([pcd_clean], 
                                          window_name="Isolated Pikachu",
                                          width=800, height=800)
        else:
            print("Skipping visualization (no display available)")
    except Exception as e:
        print(f"Skipping visualization: {e}")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)