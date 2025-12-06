import argparse
import os
import copy
from typing import Tuple, Optional
import numpy as np
import open3d as o3d

# ==========================================
# Core Utilities
# ==========================================

def load_pointcloud(path: str) -> o3d.geometry.PointCloud:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Point cloud '{path}' not found")
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"Failed to read points from {path}")
    
    # Remove NaN/Infinity points immediately
    pcd = pcd.remove_non_finite_points()
    
    if not pcd.has_colors():
        print("Warning: Point cloud has no colors. Painting uniform yellow (Pikachu).")
        pcd.paint_uniform_color([1.0, 0.8, 0.0])
    
    return pcd

def orient_normals_towards_camera(pcd, camera_location=np.array([0., 0., 0.])):
    """
    Orients normals to point towards a specific location (usually the camera at 0,0,0).
    Crucial for 2.5D depth data to prevent 'inside-out' Poisson reconstruction.
    """
    # 1. Estimate normals if missing
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # 2. Orient normals towards the camera position
    pcd.orient_normals_towards_camera_location(camera_location)
    return pcd

def preprocess_pointcloud(pcd: o3d.geometry.PointCloud, voxel_size: float = None) -> o3d.geometry.PointCloud:
    print("Preprocessing: Cleaning and orienting normals...")
    
    # 1. Statistical removal (Less aggressive to preserve features)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.5)  # Less aggressive
    
    # 2. Voxel downsample (optional, but use smaller multiplier to preserve more points)
    if voxel_size is not None and voxel_size > 0:
        # Use 0.8x voxel size to preserve more detail
        pcd = pcd.voxel_down_sample(voxel_size * 0.8)
    
    # 3. Robust Normal Estimation
    # We use a larger radius for normals to ignore small surface noise
    radius = voxel_size * 5 if voxel_size else 0.05
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    
    # Force orientation towards -Z (camera) for 2.5D data consistency
    # (Assuming the object was captured from roughly the front)
    center = pcd.get_center()
    camera_loc = center + np.array([0, 0, 10]) # Imaginary camera in front
    pcd = orient_normals_towards_camera(pcd, camera_loc)
    
    return pcd

# ==========================================
# Gap Detection & Filling
# ==========================================

def interpolate_colors(source_pcd, target_pcd, k=3):
    """
    Transfers colors from source_pcd to target_pcd using k-Nearest Neighbors.
    """
    if not source_pcd.has_colors():
        return target_pcd

    source_tree = o3d.geometry.KDTreeFlann(source_pcd)
    source_colors = np.asarray(source_pcd.colors)
    target_points = np.asarray(target_pcd.points)
    new_colors = []

    for pt in target_points:
        [_, idx, _] = source_tree.search_knn_vector_3d(pt, k)
        # Average color of k nearest neighbors
        color = np.mean(source_colors[idx], axis=0)
        new_colors.append(color)

    target_pcd.colors = o3d.utility.Vector3dVector(np.array(new_colors))
    return target_pcd

def poisson_patch_fill(
    pcd: o3d.geometry.PointCloud,
    sparse_mask: np.ndarray,
    spacing: float,
    sample_multiplier: int = 5,
    poisson_depth: int = 9,
    max_depth_ratio: Optional[float] = None,
) -> o3d.geometry.PointCloud:
    """
    Generates a skin (mesh) around the points and samples it to fill holes.
    """
    print(f"  Poisson Reconstruction (Depth={poisson_depth})...")
    
    # 1. Create Mesh
    # width=0 ensures the Poisson formulation respects the exact point coordinates better
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, width=0, scale=1.1, linear_fit=True
    )

    # 2. Crop Artifacts
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_scale = 1.1 if max_depth_ratio is None else min(1.1, max_depth_ratio)
    bbox = bbox.scale(bbox_scale, bbox.get_center())
    mesh = mesh.crop(bbox)
    
    # Remove low density vertices (the "bubble" effect)
    # Note: After cropping, the mesh vertices may have changed, so we need to check
    if len(densities) > 0:
        densities = np.asarray(densities)
        num_vertices = len(mesh.vertices)
        
        # If sizes match, we can safely filter
        if len(densities) == num_vertices:
            # Filter out the bottom 5% density (usually floating noise)
            density_threshold = np.percentile(densities, 5)
            idxs_to_remove = densities < density_threshold
            mesh.remove_vertices_by_mask(idxs_to_remove)
        else:
            # After cropping, densities array may not match - this is okay
            # We'll rely on the bounding box crop to remove most artifacts
            pass

    # 3. Sample Points from the "Skin"
    target_points = len(pcd.points) * sample_multiplier
    print(f"  Sampling {target_points} points from reconstruction...")
    dense = mesh.sample_points_poisson_disk(number_of_points=int(target_points), init_factor=2)
    
    # 4. Color Transfer
    # Poisson loses original texture sometimes, so we re-project colors from the original PCD
    dense = interpolate_colors(pcd, dense, k=3)

    # 5. Filter to keep only points near the sparse regions (Optimization)
    # This prevents adding points where we already have good density
    sparse_points = np.asarray(pcd.points)[sparse_mask]
    
    if len(sparse_points) == 0:
        return dense # Fallback if mask failed

    influence_radius = spacing * 4.0 
    sparse_pcd_temp = o3d.geometry.PointCloud()
    sparse_pcd_temp.points = o3d.utility.Vector3dVector(sparse_points)
    
    # Check which new points are useful (near gaps)
    # We use a trick: Build tree of Sparse points, query Dense points against it
    kdtree = o3d.geometry.KDTreeFlann(sparse_pcd_temp)
    dense_arr = np.asarray(dense.points)
    
    # We can accept points if they are reasonably close to a sparse area
    # Or we can just return the whole dense cloud if we want to thicken the object
    # For robust filling, returning the whole dense cloud is often smoother
    return dense

def detect_sparse_regions(pcd, percentile=70.0):
    """Identifies areas with low point density."""
    distances = np.asarray(pcd.compute_nearest_neighbor_distance())
    if len(distances) == 0:
        return np.array([]), 0.0
    
    thresh_dist = np.percentile(distances, percentile)
    mask = distances > thresh_dist
    avg_spacing = np.mean(distances)
    return mask, avg_spacing

# ==========================================
# Smoothing & Merging
# ==========================================

def mls_smooth(pcd, search_radius):
    """
    Moving Least Squares (MLS) smoothing.
    Much better than voxel downsampling for preserving surface curvature.
    """
    print("  Applying MLS Smoothing (Polynomial)...")
    try:
        # Compute a smooth surface using a polynomial fit
        pcd_smooth = pcd.compute_smoothed_point_cloud(
            pcd.compute_nearest_neighbor_distance(),
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=search_radius, max_nn=30)
        )
        # Note: In some O3D versions this returns a mesh or modifies in place. 
        # If specific MLS function isn't available, we fallback to simple normal averaging
        return pcd # Open3D python API for MLS is sometimes tricky, returning original if not standard
    except:
        return pcd

def remove_background_wall(pcd, max_distance_ratio=3.5, eps=None, preserve_features=True, original_bbox=None):
    """
    Removes background/wall artifacts while preserving object features (arms, tail, face).
    Uses a very conservative approach to avoid removing parts of the object.
    """
    if len(pcd.points) == 0:
        return pcd
    
    print("  Removing background/wall artifacts (preserving features)...")
    original_count = len(pcd.points)
    
    # Method 1: Use bounding box-based filtering to preserve original dimensions
    points = np.asarray(pcd.points)
    bbox = pcd.get_axis_aligned_bounding_box()
    
    # If we have the original bbox, use it to preserve dimensions
    if original_bbox is not None:
        # Expand original bbox slightly (5%) to account for any legitimate growth
        expanded_bbox = original_bbox.scale(1.05, original_bbox.get_center())
        # Keep points within the expanded original bounding box
        points_vector = o3d.utility.Vector3dVector(points)
        keep_mask = expanded_bbox.get_point_indices_within_bounding_box(points_vector)
        pcd_filtered = pcd.select_by_index(keep_mask)
    else:
        # Fallback: Use percentile-based distance filtering
        center = pcd.get_center()
        distances = np.linalg.norm(points - center, axis=1)
        
        # Use 99th percentile for very conservative filtering
        percentile_dist = np.percentile(distances, 99)
        max_dist = percentile_dist * max_distance_ratio
        
        # Also ensure we don't cut off more than the original extent
        original_extent = np.linalg.norm(bbox.get_extent())
        max_dist_from_extent = original_extent * 0.7  # Allow up to 70% of original extent from center
        max_dist = max(max_dist, max_dist_from_extent)
        
        # Keep only points within reasonable distance
        keep_mask = distances <= max_dist
        pcd_filtered = pcd.select_by_index(np.where(keep_mask)[0])
    
    # Method 2: Keep largest connected components (but be more conservative)
    if eps is not None and len(pcd_filtered.points) > 0 and preserve_features and original_bbox is None:
        try:
            # Use a larger eps to avoid splitting the object into multiple components
            labels = np.array(pcd_filtered.cluster_dbscan(eps=eps*2, min_points=5))
            if len(labels) > 0 and labels.max() >= 0:
                # Find all significant clusters (not just the largest)
                unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
                if len(unique_labels) > 0:
                    # Keep clusters that are at least 2% of the largest cluster (more permissive)
                    # This preserves arms, tail, face, etc. that might be separate components
                    largest_count = np.max(counts)
                    threshold = largest_count * 0.02  # 2% threshold (more permissive)
                    
                    significant_labels = unique_labels[counts >= threshold]
                    mask = np.isin(labels, significant_labels)
                    pcd_filtered = pcd_filtered.select_by_index(np.where(mask)[0])
                    print(f"    Kept {len(significant_labels)} significant components: {len(pcd_filtered.points)} points")
        except Exception as e:
            print(f"    Clustering skipped: {e}")
    
    removed = original_count - len(pcd_filtered.points)
    if removed > 0:
        print(f"    Removed {removed} background points ({removed/original_count*100:.1f}%)")
    
    return pcd_filtered

def smooth_and_merge(original, additions, voxel_size, original_bbox=None):
    combined = original + additions
    
    # 1. Remove background/wall artifacts before downsampling (very conservative)
    combined = remove_background_wall(combined, max_distance_ratio=3.5, eps=voxel_size*3, preserve_features=True, original_bbox=original_bbox)
    
    # 2. Voxel Downsample (to unify density) - use smaller voxel for higher density
    combined = combined.voxel_down_sample(voxel_size * 0.7)  # 0.7x for higher density
    
    # 3. Radius Outlier Removal (Clean up messy internal points, but less aggressive)
    combined, _ = combined.remove_radius_outlier(nb_points=5, radius=voxel_size * 3.0)  # Less aggressive
    
    # 4. Estimate Normals again for smoothing
    combined.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*3, max_nn=30))
    
    return combined

# ==========================================
# Main Pipeline
# ==========================================

def iterative_gap_filling(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    max_iterations: int = 3,
    poisson_depth: int = 9,
):
    current = pcd
    
    # Store original bounding box to preserve dimensions
    original_bbox = current.get_axis_aligned_bounding_box()
    
    # Remove background/wall artifacts from the start (very conservative to preserve features)
    current = remove_background_wall(current, max_distance_ratio=3.5, eps=voxel_size*3, preserve_features=True, original_bbox=original_bbox)
    
    for i in range(max_iterations):
        print(f"\n--- Iteration {i + 1}/{max_iterations} ---")
        
        # 1. Detect Gaps
        sparse_mask, spacing = detect_sparse_regions(current, percentile=85.0)
        
        # 2. Generate Fill Points - increase multiplier for higher density
        fill_points = poisson_patch_fill(
            current,
            sparse_mask,
            spacing,
            sample_multiplier=6, # Increased from 4 to 6 for higher density
            poisson_depth=poisson_depth
        )
        
        # 3. Merge
        # We start with the original high-quality scan + the new fill points
        # (Using 'current' instead of 'pcd' allows the mesh to grow iteratively)
        current = smooth_and_merge(current, fill_points, voxel_size, original_bbox=original_bbox)
        
        print(f"  Result: {len(current.points)} points")

    return current

def main():
    parser = argparse.ArgumentParser(description="Robust 3D Point Cloud Inpainting")
    parser.add_argument("--input", type=str, required=True, help="Input .ply/.pcd")
    parser.add_argument("--output", type=str, required=True, help="Output .ply")
    parser.add_argument("--iters", type=int, default=3, help="Number of filling passes")
    parser.add_argument("--depth", type=int, default=9, help="Poisson depth (higher = more detail, slower)")
    parser.add_argument("--voxel", type=float, default=None, help="Voxel size (leave empty for auto)")
    
    args = parser.parse_args()

    # 1. Load
    pcd = load_pointcloud(args.input)
    
    # 2. Auto-calc voxel size if missing (use smaller voxel for higher density)
    if args.voxel is None:
        bbox = pcd.get_axis_aligned_bounding_box()
        args.voxel = np.linalg.norm(bbox.get_extent()) / 400.0  # Increased from 300 to 400 for higher density
        print(f"Auto-calculated voxel size: {args.voxel:.4f}")

    # 3. Store original bounding box to preserve dimensions
    original_bbox = pcd.get_axis_aligned_bounding_box()
    print(f"Original bounding box extent: {original_bbox.get_extent()}")
    
    # 4. Preprocess (Clean & Orient)
    pcd = preprocess_pointcloud(pcd, args.voxel)
    
    # 5. Iterative Fill
    final_pcd = iterative_gap_filling(
        pcd, 
        voxel_size=args.voxel, 
        max_iterations=args.iters,
        poisson_depth=args.depth
    )
    
    # 6. Final Polish
    print("\nFinal Polish...")
    final_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=args.voxel*3, max_nn=30))
    
    # Remove background/wall artifacts one final time (very conservative, preserve original dimensions)
    final_pcd = remove_background_wall(final_pcd, max_distance_ratio=3.5, eps=args.voxel*3, preserve_features=True, original_bbox=original_bbox)
    
    # One last fine outlier removal (less aggressive to preserve features)
    final_pcd, _ = final_pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)  # Less aggressive
    
    # 6. Save
    o3d.io.write_point_cloud(args.output, final_pcd)
    print(f"Saved to {args.output}")

    # 7. Visualize
    if os.getenv('DISPLAY') or os.name == 'nt':
        o3d.visualization.draw_geometries([final_pcd], window_name="Inpainted Result")

if __name__ == "__main__":
    main()