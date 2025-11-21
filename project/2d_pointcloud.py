import torch
from PIL import Image
import numpy as np
import open3d as o3d

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load MiDaS depth model
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform  # for DPT_Large / DPT_Hybrid

def get_depth_from_image(image):
    # Ensure image is RGB numpy array
    if isinstance(image, Image.Image):
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_np = np.array(image)
    else:
        image_np = image
        # Ensure it's RGB (3 channels)
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]  # Remove alpha channel
    
    original_size = image_np.shape[:2]  # (height, width)
    
    # Apply transform - transform expects numpy array and returns a tensor
    input_tensor = transform(image_np)
    
    # Handle if transform returns a dict (some versions do)
    if isinstance(input_tensor, dict):
        input_tensor = input_tensor["image"]
    
    input_tensor = input_tensor.to(device)
    
    # Add batch dimension if needed [C, H, W] -> [1, C, H, W]
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        depth = midas(input_tensor)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=original_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth = depth.cpu().numpy()
    d_min, d_max = depth.min(), depth.max()
    depth_visual = (depth - d_min) / (d_max - d_min + 1e-8)
    return depth, depth_visual


def depth_to_pointcloud(pil_img, depth, fx=None, fy=None, cx=None, cy=None, depth_scale=1.0):
    """
    pil_img: PIL.Image (same size as depth)
    depth:  (H, W) numpy array with depth values (relative is fine)
    fx, fy: focal lengths (if None, we choose simple defaults)
    cx, cy: principal point (if None, center of image)
    depth_scale: multiply depth by this if you want to rescale
    """
    img_np = np.array(pil_img).astype(np.float32) / 255.0   # H, W, 3
    H, W = depth.shape

    if fx is None: fx = max(H, W)
    if fy is None: fy = max(H, W)
    if cx is None: cx = W / 2.0
    if cy is None: cy = H / 2.0

    Z = depth * depth_scale
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy

    # Flatten
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    Z = Z.reshape(-1)
    colors = img_np.reshape(-1, 3)

    # Filter invalid depths
    valid = np.isfinite(Z) & (Z > 0)
    points = np.stack([X[valid], -Y[valid], Z[valid]], axis=1)  # -Y to match usual camera coords
    colors = colors[valid]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


if __name__ == "__main__":
    img = Image.open("results/test_images/pikachu_inpainted.png").convert("RGB")
    depth, depth_vis = get_depth_from_image(img)
    print(depth.shape)
    print(depth_vis.shape)
    # Save a visualization if you want
    depth_vis_img = Image.fromarray((depth_vis * 255).astype(np.uint8))
    depth_vis_img.save("results/test_images/pikachu_depth_vis.png")

    pcd = depth_to_pointcloud(img, depth)

    # Save to file
    o3d.io.write_point_cloud("results/test_images/pikachu_pointcloud.ply", pcd)
    print(f"Point cloud saved to results/test_images/pikachu_pointcloud.ply")
    print(f"Point cloud has {len(pcd.points)} points")
    
    # Visualize (only if display is available)
    try:
        import os
        if os.getenv('DISPLAY') is not None:
            o3d.visualization.draw_geometries([pcd])
        else:
            print("Skipping visualization (no display available)")
    except Exception as e:
        print(f"Skipping visualization: {e}")

