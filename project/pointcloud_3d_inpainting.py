import argparse
import os
from typing import Tuple, Optional

import numpy as np
import open3d as o3d


def load_pointcloud(path: str) -> o3d.geometry.PointCloud:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Point cloud '{path}' not found")
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise ValueError(f"Failed to read points from {path}")
    if not pcd.has_colors():
        pcd.paint_uniform_color([1.0, 1.0, 1.0])
    return pcd


def preprocess_pointcloud(
    pcd: o3d.geometry.PointCloud, voxel_size: float = None
) -> o3d.geometry.PointCloud:
    if voxel_size is None:
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = np.linalg.norm(bbox.get_extent())
        voxel_size = extent / 400.0
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 6, max_nn=50)
    )
    pcd.orient_normals_consistent_tangent_plane(50)
    return pcd


def detect_sparse_regions(
    pcd: o3d.geometry.PointCloud, percentile: float = 70.0, method: str = "combined"
) -> Tuple[np.ndarray, float]:
    """
    Detect sparse regions in point cloud using multiple methods.
    
    Args:
        pcd: Input point cloud
        percentile: Percentile threshold for distance-based detection
        method: Detection method - "distance", "density", or "combined"
    
    Returns:
        Tuple of (sparse_mask, avg_sparse_spacing)
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise RuntimeError("Point cloud is empty.")
    
    # Method 1: Nearest neighbor distance
    distances = np.asarray(pcd.compute_nearest_neighbor_distance())
    if len(distances) == 0:
        raise RuntimeError("Could not compute nearest neighbor distances.")
    
    thresh_dist = np.percentile(distances, percentile)
    mask_dist = distances > thresh_dist
    
    if method == "distance":
        avg_sparse_spacing = float(np.mean(distances[mask_dist])) if mask_dist.any() else float(np.mean(distances))
        return mask_dist, avg_sparse_spacing
    
    # Method 2: Density-based (using k-nearest neighbors)
    k = min(30, len(points) - 1)
    if k > 0:
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        densities = []
        for i in range(len(points)):
            [count, _, _] = kdtree.search_knn_vector_3d(points[i], k + 1)
            # Average distance to k nearest neighbors (inverse density)
            if count > 1:
                densities.append(distances[i])
            else:
                densities.append(np.max(distances))
        densities = np.array(densities)
        thresh_density = np.percentile(densities, percentile)
        mask_density = densities > thresh_density
    else:
        mask_density = mask_dist
    
    if method == "density":
        avg_sparse_spacing = float(np.mean(distances[mask_density])) if mask_density.any() else float(np.mean(distances))
        return mask_density, avg_sparse_spacing
    
    # Method 3: Combined (union of both methods)
    mask_combined = mask_dist | mask_density
    avg_sparse_spacing = float(np.mean(distances[mask_combined])) if mask_combined.any() else float(np.mean(distances))
    return mask_combined, avg_sparse_spacing


def apply_depth_constraints(
    pcd: o3d.geometry.PointCloud,
    max_depth_ratio: float = 1.3,
    center: Optional[np.ndarray] = None,
    extent: Optional[float] = None,
) -> o3d.geometry.PointCloud:
    """
    Filter out points that extend too far from the main object (e.g., prevent tail from being too deep).
    
    Args:
        pcd: Input point cloud
        max_depth_ratio: Maximum allowed distance from center as ratio of bounding box extent (default: 1.3)
        center: Optional center point (if None, uses bounding box center)
        extent: Optional extent value (if None, computes from bounding box)
    
    Returns:
        Filtered point cloud
    """
    if len(pcd.points) == 0:
        return pcd
    
    points = np.asarray(pcd.points)
    bbox = pcd.get_axis_aligned_bounding_box()
    
    if center is None:
        center = bbox.get_center()
    if extent is None:
        extent = np.linalg.norm(bbox.get_extent())
    
    # Calculate distances from center
    distances = np.linalg.norm(points - center, axis=1)
    max_distance = extent * max_depth_ratio
    
    # Keep points within the depth constraint
    keep_mask = distances <= max_distance
    filtered_pcd = pcd.select_by_index(np.where(keep_mask)[0])
    
    removed_count = len(points) - len(filtered_pcd.points)
    if removed_count > 0:
        print(f"Removed {removed_count} points exceeding depth constraint (max_depth_ratio={max_depth_ratio:.2f})")
    
    return filtered_pcd


def poisson_patch_fill(
    pcd: o3d.geometry.PointCloud,
    sparse_mask: np.ndarray,
    spacing: float,
    sample_multiplier: int = 5,
    poisson_depth: int = 10,
    max_depth_ratio: Optional[float] = None,
) -> o3d.geometry.PointCloud:
    """
    Fill sparse regions using Poisson surface reconstruction.
    
    Args:
        pcd: Input point cloud
        sparse_mask: Boolean mask indicating sparse regions
        spacing: Average spacing in sparse regions
        sample_multiplier: Multiplier for target number of points
        poisson_depth: Depth parameter for Poisson reconstruction
        max_depth_ratio: Optional depth constraint ratio (if None, no constraint applied)
    
    Returns:
        Point cloud with filled regions
    """
    print("Running Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=poisson_depth, linear_fit=True
    )

    bbox = pcd.get_axis_aligned_bounding_box()
    # Use tighter bounding box to prevent excessive extension
    bbox_scale = 1.05 if max_depth_ratio is None else min(1.05, max_depth_ratio * 0.8)
    bbox = bbox.scale(bbox_scale, bbox.get_center())
    mesh = mesh.crop(bbox)

    target_points = max(len(pcd.points) * sample_multiplier, len(pcd.points) + 1000)
    dense = mesh.sample_points_poisson_disk(number_of_points=target_points, init_factor=5)

    sparse_points = np.asarray(pcd.points)[sparse_mask]
    if len(sparse_points) == 0:
        print("No sparse regions detected; returning densified cloud.")
        if max_depth_ratio is not None:
            dense = apply_depth_constraints(dense, max_depth_ratio=max_depth_ratio)
        return dense

    # Use larger influence radius for better gap filling
    influence_radius = spacing * 3.0  # Increased from 2.5
    sparse_pcd = o3d.geometry.PointCloud()
    sparse_pcd.points = o3d.utility.Vector3dVector(sparse_points)
    kdtree = o3d.geometry.KDTreeFlann(sparse_pcd)

    dense_arr = np.asarray(dense.points)
    keep_mask = np.zeros(len(dense_arr), dtype=bool)
    for idx, pt in enumerate(dense_arr):
        [k, _, dist2] = kdtree.search_knn_vector_3d(pt, 1)
        if k > 0 and np.sqrt(dist2[0]) < influence_radius:
            keep_mask[idx] = True
    kept = dense.select_by_index(np.where(keep_mask)[0])
    
    # Apply depth constraints if specified
    if max_depth_ratio is not None:
        kept = apply_depth_constraints(kept, max_depth_ratio=max_depth_ratio)
    
    print(f"Generated {len(kept.points)} candidate fill points near sparse regions.")
    return kept


def smooth_and_merge(
    original: o3d.geometry.PointCloud,
    additions: o3d.geometry.PointCloud,
    voxel_size: float,
    max_depth_ratio: Optional[float] = None,
) -> o3d.geometry.PointCloud:
    """
    Merge and smooth point clouds with optional depth constraints.
    
    Args:
        original: Original point cloud
        additions: New points to add
        voxel_size: Voxel size for downsampling
        max_depth_ratio: Optional depth constraint ratio
    
    Returns:
        Merged and smoothed point cloud
    """
    combined = original + additions
    print(f"Combined cloud has {len(combined.points)} points before smoothing.")
    
    # Apply depth constraints before merging if specified
    if max_depth_ratio is not None:
        combined = apply_depth_constraints(combined, max_depth_ratio=max_depth_ratio)
    
    # Use less aggressive downsampling to preserve more points and reduce gaps
    combined = combined.voxel_down_sample(voxel_size * 0.7)  # Changed from 0.8 to 0.7
    combined.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 6, max_nn=60)
    )
    # Less aggressive outlier removal to preserve more points
    combined, _ = combined.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.5)  # Changed from 2.0 to 2.5
    return combined


def iterative_gap_filling(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    max_iterations: int = 3,
    sparse_percentile: float = 70.0,
    poisson_depth: int = 10,
    max_depth_ratio: Optional[float] = None,
    min_improvement: float = 0.05,
) -> o3d.geometry.PointCloud:
    """
    Iteratively fill gaps in point cloud until convergence.
    
    Args:
        pcd: Input point cloud
        voxel_size: Voxel size for processing
        max_iterations: Maximum number of iterations
        sparse_percentile: Percentile for sparse region detection
        poisson_depth: Depth for Poisson reconstruction
        max_depth_ratio: Optional depth constraint ratio
        min_improvement: Minimum improvement in point count to continue
    
    Returns:
        Point cloud with gaps filled
    """
    current = pcd
    prev_point_count = len(current.points)
    
    for iteration in range(max_iterations):
        print(f"\n=== Iteration {iteration + 1}/{max_iterations} ===")
        
        # Detect sparse regions
        sparse_mask, spacing = detect_sparse_regions(current, percentile=sparse_percentile, method="combined")
        sparse_ratio = sparse_mask.sum() / len(sparse_mask)
        print(f"Sparse region fraction: {sparse_ratio:.2%}, avg spacing {spacing:.4f}")
        
        # If sparse regions are minimal, we're done
        if sparse_ratio < 0.01:
            print("Sparse regions are minimal, stopping iterations.")
            break
        
        # Fill sparse regions
        fill_points = poisson_patch_fill(
            current,
            sparse_mask,
            spacing,
            sample_multiplier=5,
            poisson_depth=poisson_depth,
            max_depth_ratio=max_depth_ratio,
        )
        
        # Merge and smooth
        current = smooth_and_merge(current, fill_points, voxel_size, max_depth_ratio=max_depth_ratio)
        
        # Check improvement
        new_point_count = len(current.points)
        improvement = (new_point_count - prev_point_count) / max(prev_point_count, 1)
        print(f"Point count: {prev_point_count} -> {new_point_count} (improvement: {improvement:.2%})")
        
        if improvement < min_improvement:
            print(f"Improvement below threshold ({min_improvement:.2%}), stopping iterations.")
            break
        
        prev_point_count = new_point_count
    
    return current


def inpaint_pointcloud_3d(
    input_path: str,
    output_path: str,
    voxel_size: float = None,
    sparse_percentile: float = 70.0,
    poisson_depth: int = 10,
    max_depth_ratio: float = 1.3,
    use_iterative: bool = True,
    max_iterations: int = 3,
):
    """
    Inpaint point cloud with improved gap filling and depth constraints.
    
    Args:
        input_path: Path to input point cloud
        output_path: Path to save output point cloud
        voxel_size: Optional voxel size for preprocessing
        sparse_percentile: Percentile for sparse region detection
        poisson_depth: Depth parameter for Poisson reconstruction
        max_depth_ratio: Maximum depth ratio to constrain tail/extensions (default: 1.3)
                        Set to None to disable depth constraints
        use_iterative: Whether to use iterative gap filling (default: True)
        max_iterations: Maximum iterations for iterative filling
    """
    pcd = load_pointcloud(input_path)
    print(f"Loaded {len(pcd.points)} points from {input_path}")
    pcd = preprocess_pointcloud(pcd, voxel_size)
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = np.linalg.norm(bbox.get_extent())
    voxel_used = extent / 400.0 if voxel_size is None else voxel_size
    
    print(f"Point cloud extent: {extent:.4f}")
    if max_depth_ratio is not None:
        print(f"Depth constraint: max_depth_ratio={max_depth_ratio:.2f}")

    if use_iterative:
        # Use iterative gap filling for better results
        result = iterative_gap_filling(
            pcd,
            voxel_size=voxel_used,
            max_iterations=max_iterations,
            sparse_percentile=sparse_percentile,
            poisson_depth=poisson_depth,
            max_depth_ratio=max_depth_ratio,
        )
    else:
        # Single-pass filling (original behavior)
        sparse_mask, spacing = detect_sparse_regions(pcd, percentile=sparse_percentile, method="combined")
        sparse_ratio = sparse_mask.sum() / len(sparse_mask)
        print(f"Sparse region fraction: {sparse_ratio:.2%}, avg spacing {spacing:.4f}")

        fill_points = poisson_patch_fill(
            pcd,
            sparse_mask,
            spacing,
            sample_multiplier=5,
            poisson_depth=poisson_depth,
            max_depth_ratio=max_depth_ratio,
        )

        result = smooth_and_merge(pcd, fill_points, voxel_used, max_depth_ratio=max_depth_ratio)
    
    # Final depth constraint pass to ensure no points exceed limits
    if max_depth_ratio is not None:
        result = apply_depth_constraints(result, max_depth_ratio=max_depth_ratio)
    
    o3d.io.write_point_cloud(output_path, result)
    print(f"\nWrote filled point cloud with {len(result.points)} points to '{output_path}'")


def parse_args():
    parser = argparse.ArgumentParser(description="Direct 3D point cloud inpainting via Poisson reconstruction.")
    parser.add_argument("--input", type=str, required=True, help="Input incomplete point cloud (.ply/.pcd)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the filled point cloud")
    parser.add_argument("--voxel_size", type=float, default=None, help="Optional voxel size for preprocessing")
    parser.add_argument("--sparse_percentile", type=float, default=70.0, help="Percentile to mark sparse points")
    parser.add_argument("--poisson_depth", type=int, default=10, help="Poisson reconstruction depth")
    parser.add_argument("--max_depth_ratio", type=float, default=1.3, help="Maximum depth ratio to constrain extensions (e.g., tail). Set to 0 to disable.")
    parser.add_argument("--no_iterative", action="store_true", help="Disable iterative gap filling (use single pass)")
    parser.add_argument("--max_iterations", type=int, default=3, help="Maximum iterations for iterative filling")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    max_depth_ratio = args.max_depth_ratio if args.max_depth_ratio > 0 else None
    inpaint_pointcloud_3d(
        input_path=args.input,
        output_path=args.output,
        voxel_size=args.voxel_size,
        sparse_percentile=args.sparse_percentile,
        poisson_depth=args.poisson_depth,
        max_depth_ratio=max_depth_ratio,
        use_iterative=not args.no_iterative,
        max_iterations=args.max_iterations,
    )

