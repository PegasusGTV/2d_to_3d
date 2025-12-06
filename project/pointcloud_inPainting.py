import torch
import numpy as np
from pytorch3d.structures import Pointclouds
import open3d as o3d
from PIL import Image
import math
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)
from pytorch3d.renderer.cameras import look_at_view_transform
from diffusers import StableDiffusionInpaintPipeline
from typing import Optional

# Import depth estimation functions from 2d_pointcloud.py
import sys
import os
from importlib import import_module

# Add current directory to path to import 2d_pointcloud
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import functions using importlib (needed because module name starts with number)
try:
    # Use importlib to import module with numeric prefix
    pointcloud_module = import_module('2d_pointcloud')
    get_depth_from_image = pointcloud_module.get_depth_from_image
    depth_to_pointcloud = pointcloud_module.depth_to_pointcloud
except ImportError as e:
    # Fallback: define functions here if import fails
    print(f"Warning: Could not import from 2d_pointcloud.py: {e}")
    print("Depth estimation may not work properly")
    def get_depth_from_image(image):
        raise NotImplementedError("Please ensure 2d_pointcloud.py is available")
    def depth_to_pointcloud(*args, **kwargs):
        raise NotImplementedError("Please ensure 2d_pointcloud.py is available")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def o3d_to_pytorch3d(pcd):
    """Convert Open3D point cloud to PyTorch3D format"""
    pts = np.asarray(pcd.points).astype(np.float32)   # (N, 3)
    cols = np.asarray(pcd.colors).astype(np.float32)  # (N, 3) in [0,1]

    pts = torch.from_numpy(pts)[None].to(device)      # (1, N, 3)
    cols = torch.from_numpy(cols)[None].to(device)    # (1, N, 3)

    return Pointclouds(points=pts, features=cols).to(device)

def make_points_renderer(image_size=256):
    """Create a point cloud renderer"""
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=0.01,
        points_per_pixel=10,
    )
    rasterizer = PointsRasterizer(raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(),
    )
    return renderer.to(device)

def make_camera_pose(angle_deg, radius=1.8):
    """Create camera pose for a given viewing angle"""
    R, T = look_at_view_transform(
        dist=radius,
        elev=10.0,
        azim=angle_deg,
    )
    return R.to(device), T.to(device)

def render_view(pcd_torch, angle_deg, image_size=256):
    """Render a point cloud from a specific viewing angle"""
    R, T = make_camera_pose(angle_deg, radius=1.8)

    cameras = FoVPerspectiveCameras(
        R=R, T=T, device=device
    )

    images = renderer(pcd_torch, cameras=cameras)  # (1, H, W, 3) or (1, H, W, 4)
    img = images[0, ..., :3].detach().cpu().numpy()   # RGB
    
    # Check if alpha channel exists
    if images.shape[-1] >= 4:
        alpha = images[0, ..., 3].detach().cpu().numpy()  # alpha
        # Holes where no point was projected (low alpha or black pixels)
        holes = (alpha < 1e-3) | (img.sum(axis=2) < 1e-3)
    else:
        # If no alpha channel, detect holes as black/dark pixels
        holes = img.sum(axis=2) < 1e-3  # Sum of RGB channels is very small

    return img, holes, R, T

def dilate_mask(mask_bool, iterations=1):
    """Simple binary dilation to slightly expand hole masks"""
    mask = mask_bool.astype(bool)
    if iterations <= 0:
        return mask
    h, w = mask.shape
    for _ in range(iterations):
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        expanded = np.zeros_like(mask, dtype=bool)
        for dx in range(3):
            for dy in range(3):
                expanded |= padded[dx:dx + h, dy:dy + w]
        mask = expanded
    return mask

def numpy_to_pil_image(img_np):
    """Convert numpy array to PIL Image"""
    img_uint8 = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(img_uint8)

def mask_bool_to_pil(mask_bool):
    """Convert boolean mask to PIL Image"""
    return Image.fromarray(mask_bool.astype(np.uint8) * 255)

def inpaint_rendered_view(
    img_np,
    hole_mask,
    prompt="a realistic photo of the object",
    negative_prompt: Optional[str] = None,
    num_inference_steps=40,
    guidance_scale=8.0,
):
    """Inpaint holes in a rendered view using Stable Diffusion"""
    img_pil = numpy_to_pil_image(img_np)
    mask_pil = mask_bool_to_pil(hole_mask)

    # Stable Diffusion works best at 512x512. Upsample inputs for inpainting.
    original_size = img_pil.size
    if img_pil.size != (512, 512):
        img_pil = img_pil.resize((512, 512), Image.BICUBIC)
        mask_pil = mask_pil.resize((512, 512), Image.NEAREST)

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=img_pil,
        mask_image=mask_pil,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    ).images[0]

    # Downsample back to the renderer resolution so that camera intrinsics stay valid
    if out.size != original_size:
        out = out.resize(original_size, Image.BICUBIC)

    return out  # PIL.Image

def backproject_inpainted_view(
    inpainted_img,
    depth,
    R,
    T,
    image_size=256,
    radius=1.8,
    depth_scale_factor=1.0,
    hole_mask=None,
    mask_dilate_iters=1,
):
    """
    Back-project an inpainted 2D view to 3D space using depth and camera parameters.
    
    Args:
        inpainted_img: PIL Image of the inpainted view
        depth: numpy array (H, W) with depth values
        R: rotation matrix (1, 3, 3) tensor
        T: translation vector (1, 3) tensor
        image_size: size of the rendered image
        radius: camera distance from origin
        depth_scale_factor: factor to scale depth values (default: 1.0)
    
    Returns:
        Open3D point cloud with new points
    """
    # Convert PIL to numpy
    img_np = np.array(inpainted_img).astype(np.float32) / 255.0  # (H, W, 3)
    H, W = depth.shape

    if hole_mask is not None:
        mask_np = np.asarray(hole_mask).astype(bool)
        if mask_np.shape != (H, W):
            mask_np = np.array(
                Image.fromarray(mask_np.astype(np.uint8) * 255).resize((W, H), Image.Nearest)
            ) > 0
        mask_np = dilate_mask(mask_np, iterations=mask_dilate_iters)
    else:
        mask_np = np.ones((H, W), dtype=bool)
    
    # Camera intrinsics (assuming FoV perspective camera)
    # For FoVPerspectiveCameras, we need to estimate focal length
    # Using a reasonable default based on image size
    fx = fy = image_size * 0.7  # Approximate focal length
    cx = W / 2.0
    cy = H / 2.0
    
    # Create pixel coordinates
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    
    # Convert depth to actual distance
    # MiDaS gives inverse depth (closer = larger values), so we invert it
    # Normalize depth values
    depth_normalized = depth.copy()
    if depth_normalized.max() > depth_normalized.min():
        depth_normalized = (depth_normalized - depth_normalized.min()) / (depth_normalized.max() - depth_normalized.min() + 1e-8)
    
    # Convert normalized depth to actual distance
    # Use a range that makes sense for the camera distance
    min_depth = radius * 0.3  # Minimum depth (closer to camera)
    max_depth = radius * 2.0   # Maximum depth (farther from camera)
    Z = min_depth + depth_normalized * (max_depth - min_depth) * depth_scale_factor
    
    # Back-project to camera coordinates
    X_cam = (xs - cx) * Z / fx
    Y_cam = (ys - cy) * Z / fy
    
    # Stack camera coordinates
    points_cam = np.stack([X_cam.reshape(-1), Y_cam.reshape(-1), Z.reshape(-1)], axis=1)
    colors = img_np.reshape(-1, 3)
    
    # Filter invalid depths
    valid = np.isfinite(points_cam).all(axis=1) & (Z.reshape(-1) > 0)
    valid &= mask_np.reshape(-1)
    points_cam = points_cam[valid]
    colors = colors[valid]
    
    # Convert to world coordinates using camera pose
    # PyTorch3D's look_at_view_transform returns world-to-camera transformation
    # So we need to invert it to get camera-to-world
    R_np = R[0].detach().cpu().numpy()  # (3, 3) - world-to-camera rotation
    T_np = T[0].detach().cpu().numpy()  # (3,) - world-to-camera translation
    
    # Invert transformation: camera-to-world
    R_inv = R_np.T  # Inverse of rotation matrix (transpose for orthonormal)
    T_inv = -R_inv @ T_np  # Inverted translation
    
    # Transform from camera to world coordinates
    points_world = (R_inv @ points_cam.T).T + T_inv
    
    # Create Open3D point cloud
    pcd_new = o3d.geometry.PointCloud()
    pcd_new.points = o3d.utility.Vector3dVector(points_world)
    pcd_new.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd_new

def estimate_pointcloud_scale(pcd):
    """Estimate the scale of a point cloud for depth scaling"""
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return 1.0
    
    # Calculate bounding box diagonal
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    scale = np.linalg.norm(extent)
    
    # Also calculate mean distance from origin
    mean_dist = np.mean(np.linalg.norm(points, axis=1))
    
    # Use average of these as scale estimate
    estimated_scale = (scale + mean_dist * 2) / 3.0
    
    return max(estimated_scale, 0.1)  # Ensure minimum scale

def fill_pointcloud_holes(pcd, voxel_size=0.01):
    """
    Fill holes in a point cloud using various Open3D techniques.
    
    Args:
        pcd: Open3D point cloud
        voxel_size: voxel size for downsampling and reconstruction
    
    Returns:
        Filled point cloud
    """
    print(f"Original point cloud has {len(pcd.points)} points")
    
    # Remove statistical outliers
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"After outlier removal: {len(pcd.points)} points")
    
    # Estimate normals if not present
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(100)
    
    # Poisson surface reconstruction to fill holes
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9, width=0, scale=1.1, linear_fit=False
        )
        
        # Remove low density vertices (likely from holes)
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Convert mesh back to point cloud
        pcd_filled = mesh.sample_points_uniformly(number_of_points=len(pcd.points) * 2)
        
        # Merge with original
        pcd_merged = pcd + pcd_filled
        print(f"After Poisson reconstruction: {len(pcd_merged.points)} points")
        
    except Exception as e:
        print(f"Poisson reconstruction failed: {e}, using original point cloud")
        pcd_merged = pcd
    
    # Remove duplicates using voxel downsampling
    pcd_final = pcd_merged.voxel_down_sample(voxel_size=voxel_size)
    print(f"After voxel downsampling: {len(pcd_final.points)} points")
    
    return pcd_final

def complete_pointcloud_inpainting(
    pcd_path,
    output_path=None,
    num_views=8,
    image_size=256,
    prompt: Optional[str] = None,
    negative_prompt: Optional[str] = "blurry, distorted, text, watermark, background, multiple objects, human",
    mask_dilate_iters: int = 2,
    num_inference_steps: int = 40,
    guidance_scale: float = 8.0,
):
    """
    Complete pipeline for point cloud inpainting:
    1. Load point cloud
    2. Render multiple views
    3. Detect and inpaint holes in 2D views
    4. Convert inpainted views back to 3D
    5. Merge with original point cloud
    6. Fill remaining holes
    
    Args:
        pcd_path: path to input point cloud file
        output_path: path to save output point cloud (optional)
        num_views: number of views to render (default 8)
        image_size: size of rendered images (default 256)
    """
    # Load point cloud
    print(f"Loading point cloud from {pcd_path}...")
    pcd = o3d.io.read_point_cloud(pcd_path)
    if len(pcd.points) == 0:
        raise ValueError(f"Failed to load point cloud from {pcd_path}")
    
    print(f"Loaded point cloud with {len(pcd.points)} points")
    
    # Ensure point cloud has colors
    if not pcd.has_colors():
        # Assign default white color
        pcd.paint_uniform_color([1.0, 1.0, 1.0])
    
    # Convert to PyTorch3D format
    pcd_torch = o3d_to_pytorch3d(pcd)
    
    # Initialize renderer
    global renderer
    renderer = make_points_renderer(image_size=image_size)
    
    # Initialize inpainting pipeline
    global pipe
    print("Loading Stable Diffusion inpainting model...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)

    if prompt is None:
        prompt = "a bright yellow Pikachu character, 3d toy, high detail, smooth lighting"
    
    # Render views and detect holes
    print(f"Rendering {num_views} views...")
    imgs = []
    masks = []
    camera_poses = []
    angles = np.linspace(0, 360, num_views, endpoint=False)
    
    for angle in angles:
        img_np, hole_mask, R, T = render_view(pcd_torch, angle_deg=angle, image_size=image_size)
        imgs.append(img_np)
        masks.append(hole_mask)
        camera_poses.append((R, T))
    
    # Inpaint views with holes
    print("Inpainting views with holes...")
    filled_views = []
    filled_view_poses = []
    filled_view_masks = []
    filled_view_angles = []
    
    for i, (img_np, mask, (R, T), angle) in enumerate(zip(imgs, masks, camera_poses, angles)):
        if mask.any():
            print(f"  Inpainting view {i+1}/{num_views} (angle: {angle:.1f}°)...")
            filled = inpaint_rendered_view(
                img_np,
                mask,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            filled_views.append(filled)
            filled_view_poses.append((R, T))
            filled_view_masks.append(mask)
            filled_view_angles.append(angle)
        else:
            print(f"  View {i+1}/{num_views} has no holes, skipping...")
    
    # Estimate point cloud scale for depth scaling
    pcd_scale = estimate_pointcloud_scale(pcd)
    print(f"Estimated point cloud scale: {pcd_scale:.3f}")
    
    # Convert inpainted views back to 3D
    print(f"Converting {len(filled_views)} inpainted views back to 3D...")
    new_pointclouds = []
    
    for i, (filled_img, mask_np, (R, T), angle) in enumerate(
        zip(filled_views, filled_view_masks, filled_view_poses, filled_view_angles)
    ):
        print(f"  Processing view {i+1}/{len(filled_views)}...")
        
        # Estimate depth from inpainted image
        depth, _ = get_depth_from_image(filled_img)
        
        # Calculate depth scale factor based on point cloud scale
        depth_scale_factor = pcd_scale / 2.0  # Adjust based on scale
        
        # Back-project to 3D
        pcd_new = backproject_inpainted_view(
            filled_img,
            depth,
            R,
            T,
            image_size=image_size,
            radius=1.8,
            depth_scale_factor=depth_scale_factor,
            hole_mask=mask_np,
            mask_dilate_iters=mask_dilate_iters,
        )
        new_pointclouds.append(pcd_new)
        print(f"    Generated {len(pcd_new.points)} new points")
    
    # Merge all point clouds
    print("Merging point clouds...")
    pcd_merged = pcd
    for pcd_new in new_pointclouds:
        pcd_merged = pcd_merged + pcd_new
    
    print(f"Merged point cloud has {len(pcd_merged.points)} points")
    
    # Fill remaining holes
    print("Filling remaining holes...")
    pcd_final = fill_pointcloud_holes(pcd_merged, voxel_size=0.01)
    
    # Save result
    if output_path is None:
        output_path = pcd_path.replace('.ply', '_filled.ply').replace('.pcd', '_filled.pcd')
    
    o3d.io.write_point_cloud(output_path, pcd_final)
    print(f"Complete point cloud saved to {output_path}")
    print(f"Final point cloud has {len(pcd_final.points)} points")
    
    return pcd_final

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete point cloud inpainting pipeline")
    parser.add_argument("--input", type=str, default="pointcloud.ply",
                        help="Input point cloud file path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output point cloud file path (optional)")
    parser.add_argument("--num_views", type=int, default=8,
                        help="Number of views to render (default: 8)")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Size of rendered images (default: 256)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Prompt to guide Stable Diffusion inpainting")
    parser.add_argument("--negative_prompt", type=str, default=None,
                        help="Negative prompt to avoid unwanted artifacts")
    parser.add_argument("--mask_dilate_iters", type=int, default=2,
                        help="How much to dilate hole masks before back-projection")
    parser.add_argument("--num_inference_steps", type=int, default=40,
                        help="Diffusion inference steps")
    parser.add_argument("--guidance_scale", type=float, default=8.0,
                        help="Classifier-free guidance scale")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found")
        print("Available files in current directory:")
        for f in os.listdir("."):
            if f.endswith(('.ply', '.pcd')):
                print(f"  - {f}")
        sys.exit(1)
    
    # Run the pipeline
    pcd_final = complete_pointcloud_inpainting(
        pcd_path=args.input,
        output_path=args.output,
        num_views=args.num_views,
        image_size=args.image_size,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        mask_dilate_iters=args.mask_dilate_iters,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
    )
    
    print("\nPoint cloud inpainting completed successfully!")
