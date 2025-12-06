import torch
import numpy as np
import open3d as o3d
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import cv2
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

class DepthEstimator:
    def __init__(self):
        print(f"Loading Depth Anything V2 (Large) on {device}...")
        # We use the HuggingFace implementation of Depth Anything V2
        self.checkpoint = "depth-anything/Depth-Anything-V2-Large-hf"
        self.image_processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.checkpoint).to(device)
        self.model.eval()

    def estimate(self, image_pil):
        # Prepare image
        inputs = self.image_processor(images=image_pil, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image_pil.size[::-1], # PIL is (W, H), torch expects (H, W)
            mode="bicubic",
            align_corners=False,
        )

        # Normalize depth for visualization and point cloud
        depth = prediction.squeeze().cpu().numpy()
        
        # Invert if necessary (Depth Anything usually outputs disparity/inverse depth)
        # We normalize to 0-1 range for consistency
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max > depth_min:
            depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        else:
            depth_normalized = np.zeros_like(depth)
        
        return depth_normalized

def filter_flying_pixels(depth_map, threshold=0.05):
    """
    Removes pixels that exist on the sharp edge between foreground and background.
    These pixels cause the 'stretching' effect in 3D.
    """
    # Calculate gradients (change in depth)
    # This finds the "edges" in the depth map
    grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Create a mask where the change is too steep
    # Threshold needs tuning: 0.05 is a good starting point for 0-1 normalized depth
    edge_mask = magnitude < threshold
    
    return edge_mask

def depth_to_pointcloud_clean(pil_img, depth, mask=None, fx=None, fy=None, cx=None, cy=None, depth_scale=10.0):
    """
    Convert depth map to point cloud with optional mask filtering.
    
    Args:
        pil_img: PIL Image
        depth: Normalized depth map (0-1 range)
        mask: Optional boolean mask for valid pixels
        fx, fy: Focal lengths (if None, auto-calculated)
        cx, cy: Principal point (if None, image center)
        depth_scale: Scale factor for depth conversion
    
    Returns:
        Open3D point cloud
    """
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    H, W = depth.shape

    # Simple intrinsic guess (approx 50mm lens equivalent)
    if fx is None: fx = max(H, W) * 1.2
    if fy is None: fy = max(H, W) * 1.2
    if cx is None: cx = W / 2.0
    if cy is None: cy = H / 2.0

    # Invert depth for 3D projection (Standard for disparity-based models)
    # 0 = far, 1 = close. We want actual Z distance.
    # We add a small epsilon to avoid division by zero
    Z = depth_scale / (depth + 0.1) 

    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy

    # Flatten
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z.reshape(-1)
    colors = img_np.reshape(-1, 3)
    
    # Flatten mask if it exists
    if mask is not None:
        valid_mask = mask.reshape(-1)
    else:
        valid_mask = np.ones_like(Z, dtype=bool)

    # Filter 1: Remove invalid mathematical depths
    valid = np.isfinite(Z) & (Z > 0) & valid_mask
    
    points = np.stack([X[valid], -Y[valid], -Z[valid]], axis=1) # -Y and -Z for standard View
    colors = colors[valid]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def cleanup_pointcloud(pcd):
    """
    Uses statistical outlier removal to clean up noise.
    """
    print("Running statistical outlier removal...")
    # nb_neighbors: higher = more aggressive filtering
    # std_ratio: lower = more aggressive filtering
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
    cleaned_pcd = pcd.select_by_index(ind)
    print(f"Cleaned point cloud: {len(pcd.points)} -> {len(cleaned_pcd.points)} points")
    return cleaned_pcd

if __name__ == "__main__":
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load Image
    img_path = os.path.join(script_dir, "results/test_images/pikachu_inpainted.png")
    if not os.path.exists(img_path):
        # Try alternative path
        img_path = "results/test_images/pikachu_inpainted.png"
    
    try:
        img = Image.open(img_path).convert("RGB")
        print(f"Loaded image from: {img_path}")
        print(f"Image size: {img.size}")
    except FileNotFoundError:
        # Generate dummy image if file not found
        print(f"Warning: Image not found at {img_path}, using random noise.")
        img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))

    # 2. Estimate Depth (Better Model)
    print("\n" + "="*50)
    print("Step 1: Estimating depth...")
    print("="*50)
    estimator = DepthEstimator()
    depth = estimator.estimate(img)
    print(f"Depth map shape: {depth.shape}, range: [{depth.min():.3f}, {depth.max():.3f}]")
    
    # 3. Create Edge Mask (The Secret Sauce)
    # This prevents the "rubber sheet" effect between objects
    print("\n" + "="*50)
    print("Step 2: Filtering flying pixels...")
    print("="*50)
    valid_edge_mask = filter_flying_pixels(depth, threshold=0.15)
    print(f"Valid pixels: {valid_edge_mask.sum()} / {valid_edge_mask.size} ({100*valid_edge_mask.sum()/valid_edge_mask.size:.1f}%)")

    # 4. Generate Point Cloud
    print("\n" + "="*50)
    print("Step 3: Generating point cloud...")
    print("="*50)
    pcd = depth_to_pointcloud_clean(img, depth, mask=valid_edge_mask)
    print(f"Initial point cloud: {len(pcd.points)} points")

    # 5. Post-Process Cleaning
    print("\n" + "="*50)
    print("Step 4: Cleaning point cloud...")
    print("="*50)
    pcd_clean = cleanup_pointcloud(pcd)

    # 6. Save
    print("\n" + "="*50)
    print("Step 5: Saving results...")
    print("="*50)
    output_dir = os.path.join(script_dir, "results/test_images")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "pikachu_high_quality.ply")
    o3d.io.write_point_cloud(output_path, pcd_clean)
    print(f"Saved to: {output_path}")
    print(f"Final point cloud has {len(pcd_clean.points)} points")

    # 7. Visualization
    print("\n" + "="*50)
    print("Step 6: Visualization...")
    print("="*50)
    try:
        if os.getenv('DISPLAY') is not None:
            print("Opening visualization window...")
            o3d.visualization.draw_geometries([pcd_clean], 
                                          window_name="Cleaned Point Cloud",
                                          width=800, height=600)
        else:
            print("Skipping visualization (no display available)")
    except Exception as e:
        print(f"Skipping visualization: {e}")
    
    print("\n" + "="*50)
    print("Done!")
    print("="*50)

