"""
GPU-accelerated Global Motion Compensation (GMC) using NVIDIA VPI.

Replaces the CPU-based pipeline in MOD_Functions.motion_compensate():
  1. GaussianBlur        → vpi.gaussian_filter     (CUDA)
  2. Optical Flow LK     → vpi.OpticalFlowPyrLK    (CUDA)
  3. findHomography      → RANSAC on GPU arrays     (CPU fallback – VPI RANSAC is async)
  4. warpPerspective     → vpi.Image.perspwarp      (CUDA)

Usage:
    from utils.gmc_vpi import GMC_VPI

    gmc = GMC_VPI(backend=vpi.Backend.CUDA)
    motion_mask = gmc.compute_mask(prev_frame_gray, cur_frame_gray)
"""

import numpy as np
import cv2

try:
    import vpi
    VPI_AVAILABLE = True
except ImportError:
    VPI_AVAILABLE = False
    print("[WARN] NVIDIA VPI not found – GMC_VPI will fall back to CPU (OpenCV).")


class GMC_VPI:
    """
    GPU-accelerated Global Motion Compensation.

    Pipeline per frame-pair (prev → cur):
        1. Gaussian blur (11×11)  on GPU
        2. Build image pyramids   on GPU
        3. Grid-based Optical Flow (Pyramidal LK)  on GPU
        4. Filter outliers (distance > threshold)
        5. Estimate Homography (RANSAC)
        6. Warp prev frame to cur coordinate system on GPU
        7. Compute border mask from warped corners   on GPU
        8. Absolute difference = motion residual

    Parameters
    ----------
    scale : int
        Internal processing scale factor (default 2 → 1920×1080).
    grid_w, grid_h : int
        Grid cell size for keypoint generation.
    blur_ksize : int
        Gaussian kernel size (must be odd).
    blur_sigma : float
        Gaussian sigma.  0 → auto from ksize.
    pyr_levels : int
        Number of pyramid levels for LK tracker.
    lk_window : int
        Search window size for LK.
    outlier_dist : float
        Maximum displacement to keep a point.
    ransac_thresh : float
        RANSAC reprojection threshold for findHomography.
    min_points : int
        Minimum tracked points needed – below this we skip compensation.
    backend : vpi.Backend or None
        VPI backend to use. None → CUDA if available.
    """

    def __init__(
        self,
        scale=2,
        grid_w=64,
        grid_h=48,
        blur_ksize=11,
        blur_sigma=0.0,
        pyr_levels=3,
        lk_window=15,
        outlier_dist=50.0,
        ransac_thresh=3.0,
        min_points=15,
        backend=None,
    ):
        if not VPI_AVAILABLE:
            raise RuntimeError(
                "NVIDIA VPI is required for GMC_VPI. "
                "Install it via: apt install python3-vpi  (Jetson) or the desktop deb package."
            )

        self.scale          = scale
        self.grid_w         = grid_w
        self.grid_h         = grid_h
        self.blur_ksize     = blur_ksize
        self.blur_sigma     = blur_sigma if blur_sigma > 0 else (0.3 * ((blur_ksize - 1) * 0.5 - 1) + 0.8)
        self.pyr_levels     = pyr_levels
        self.lk_window      = lk_window
        self.outlier_dist   = outlier_dist
        self.ransac_thresh  = ransac_thresh
        self.min_points     = min_points
        self.backend        = backend or vpi.Backend.CUDA

        # Reusable VPI objects (lazy-initialised on first call)
        self._lk_tracker    = None
        self._prev_pyr      = None
        self._grid_pts      = None      # np array of grid keypoints
        self._grid_shape    = None      # (cols, rows) of the grid
        self._proc_size     = None      # (W, H) of processing resolution

    # ──────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────

    def compute_mask(self, frame_prev_gray, frame_cur_gray):
        h_orig, w_orig = frame_cur_gray.shape[:2]
        proc_w = int(960 * self.scale)
        proc_h = int(540 * self.scale)

        # 1. Load original frames directly to GPU
        vpi_prev_orig = vpi.asimage(frame_prev_gray)
        vpi_cur_orig  = vpi.asimage(frame_cur_gray)

        with self.backend:
            # 2. Resize and Blur 100% on GPU (No more OpenCV resize!)
            vpi_prev = vpi_prev_orig.rescale((proc_w, proc_h), interp=vpi.Interp.CUBIC)
            vpi_cur  = vpi_cur_orig.rescale((proc_w, proc_h), interp=vpi.Interp.CUBIC)

            vpi_prev_blur = vpi_prev.gaussian_filter(self.blur_ksize, self.blur_sigma)
            vpi_cur_blur  = vpi_cur.gaussian_filter(self.blur_ksize, self.blur_sigma)

        # 3. Generate grid keypoints (CPU once, then upload)
        self._ensure_grid(proc_w, proc_h)

        # 4. Optical Flow – Pyramidal LK on GPU
        good_new, good_old, status = self._track_points(vpi_prev_blur, vpi_cur_blur)

        # 5. Filter outliers (CPU)
        good_new_filt, good_old_filt, avg_dist, motion_x, motion_y = self._filter_outliers(
            good_new, good_old, status
        )

        # 6. Estimate Affine Partial 2D (Similarity Transform) - robust for UAVs
        # Note: Kept on CPU via OpenCV because VPI's TransformEstimator only supports CPU backend anyway.
        # This avoids the overhead of wrapping keypoints into vpi.Array.
        if len(good_old_filt) < self.min_points:
            affine_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        else:
            affine_matrix, _ = cv2.estimateAffinePartial2D(
                good_new_filt, good_old_filt, maxIters=200, ransacReprojThreshold=self.ransac_thresh
            )
            if affine_matrix is None:
                affine_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
                
        # Create 3x3 matrix for interface compatibility and VPI perspwarp
        homo_matrix = np.eye(3, dtype=np.float64)
        homo_matrix[0:2, :] = affine_matrix

        homo_inv = np.linalg.inv(homo_matrix).astype(np.float32)

        # 7. Warp and generate Border Mask 100% on GPU
        # Trick: Warp a solid white image. Out-of-bounds pixels become black (0).
        white_mask = np.full((h_orig, w_orig), 255, dtype=np.uint8)
        vpi_white = vpi.asimage(white_mask)

        with self.backend:
            # Warp the frame
            vpi_warped = vpi_prev_orig.perspwarp(homo_inv.tolist(), border=vpi.Border.ZERO)
            # Warp the white mask
            vpi_warped_mask = vpi_white.perspwarp(homo_inv.tolist(), border=vpi.Border.ZERO)

        # Cache VPI image for FD5 Absdiff without pulling to CPU RAM
        self._last_vpi_warped = vpi_warped

        # 8. Read back warped result to CPU (required for returning NumPy arrays)
        with vpi_warped.rlock_cpu() as warped_data:
            compensated = np.array(warped_data, copy=True)
            
        with vpi_warped_mask.rlock_cpu() as mask_data:
            # Invert the warped white image to get the border mask (No OpenCV Polylines needed!)
            border_mask = 255 - np.array(mask_data, copy=True)

        return compensated, border_mask, avg_dist, motion_x, motion_y, homo_matrix

    def compute_fd5_mask(self, frame1_gray, frame2_gray, frame3_gray):
        """
        Replicate FD5_mask logic: two-direction compensation + average diff (100% GPU VPI version).
        """
        # Warp t-2 to t
        comp1, mask1, _, _, _, _ = self.compute_mask(frame1_gray, frame2_gray)
        vpi_comp1 = self._last_vpi_warped  # Use cached VPI image on GPU
        
        # Warp t+2 to t
        comp2, mask2, _, _, _, _ = self.compute_mask(frame3_gray, frame2_gray)
        vpi_comp2 = self._last_vpi_warped  # Use cached VPI image on GPU

        vpi_f2 = vpi.asimage(frame2_gray)
        
        with self.backend:
            # 100% GPU Absolute Difference
            vpi_diff1 = vpi_f2.absdiff(vpi_comp1)
            vpi_diff2 = vpi_f2.absdiff(vpi_comp2)

        # Read back diffs to average on CPU 
        # (VPI Python API lacks a simple add/divide function for images, so we average them in numpy quickly)
        with vpi_diff1.rlock_cpu() as d1_data, vpi_diff2.rlock_cpu() as d2_data:
            diff1 = np.array(d1_data, copy=False)
            diff2 = np.array(d2_data, copy=False)
            motion_diff = ((diff1.astype(np.float32) + diff2.astype(np.float32)) / 2.0).astype(np.uint8)

        return motion_diff

    # ──────────────────────────────────────────────────────────
    #  Internal helpers
    # ──────────────────────────────────────────────────────────

    def _ensure_grid(self, w, h):
        """Generate grid keypoints if not yet created or resolution changed."""
        if self._proc_size == (w, h) and self._grid_pts is not None:
            return

        self._proc_size = (w, h)
        cols = int(w / self.grid_w - 1)
        rows = int(h / self.grid_h - 1)
        self._grid_shape = (cols, rows)

        pts = []
        for i in range(cols):
            for j in range(rows):
                px = np.float32(i * self.grid_w + self.grid_w / 2.0)
                py = np.float32(j * self.grid_h + self.grid_h / 2.0)
                pts.append([px, py])

        self._grid_pts = np.array(pts, dtype=np.float32)

    def _track_points(self, vpi_prev, vpi_cur):
        """
        Run Pyramidal LK optical flow on grid keypoints.

        Returns
        -------
        pts_cur  : np.ndarray (N, 2) float32
        pts_prev : np.ndarray (N, 2) float32
        status   : np.ndarray (N,)   uint8
        """
        n_pts = len(self._grid_pts)

        # Create VPI keypoint arrays
        prev_pts = vpi.Array(vpi.Type.KEYPOINT_F32, capacity=n_pts)

        # Fill keypoints
        with prev_pts.lock_cpu() as data:
            for i, (px, py) in enumerate(self._grid_pts):
                data[i] = (px, py, 0, 0)  # (x, y, tracking_status, template_status)

        prev_pts.size = n_pts

        # Build pyramids
        with self.backend:
            pyr_prev = vpi.GaussianPyramid(vpi_prev, self.pyr_levels)
            pyr_cur  = vpi.GaussianPyramid(vpi_cur,  self.pyr_levels)

        # Run OpticalFlowPyrLK
        with self.backend:
            cur_pts, status_arr = vpi.optflow_pyr_lk(
                pyr_prev, pyr_cur,
                prev_pts,
                epsilon=0.01,
                num_iterations=30,
                window_size=self.lk_window,
            )

        # Read back results
        with cur_pts.rlock_cpu() as cur_data:
            pts_cur_raw = np.array([(p[0], p[1]) for p in cur_data[:cur_pts.size]], dtype=np.float32)

        with status_arr.rlock_cpu() as st_data:
            status = np.array(st_data[:status_arr.size], dtype=np.uint8)

        pts_prev_raw = self._grid_pts[:len(pts_cur_raw)].copy()

        return pts_cur_raw, pts_prev_raw, status

    def _filter_outliers(self, pts_new, pts_old, status):
        """Filter tracked points by status and displacement threshold."""
        # Keep only successfully tracked points
        valid = status == 0  # VPI: 0 = success
        pts_new = pts_new[valid]
        pts_old = pts_old[valid]

        if len(pts_new) == 0:
            return pts_new, pts_old, 0.0, 0.0, 0.0

        # Compute displacement
        dx = pts_new[:, 0] - pts_old[:, 0]
        dy = pts_new[:, 1] - pts_old[:, 1]
        dist = np.sqrt(dx ** 2 + dy ** 2)

        # Filter by maximum distance
        keep = dist < self.outlier_dist
        pts_new_f = pts_new[keep]
        pts_old_f = pts_old[keep]
        dx_f = dx[keep]
        dy_f = dy[keep]
        dist_f = dist[keep]

        avg_dist = float(np.mean(dist_f)) if len(dist_f) > 0 else 0.0
        motion_x = float(np.mean(dx_f))   if len(dx_f) > 0 else 0.0
        motion_y = float(np.mean(dy_f))   if len(dy_f) > 0 else 0.0

        return pts_new_f, pts_old_f, avg_dist, motion_x, motion_y

    def _compute_border_mask(self, homo_matrix, w, h):
        """Compute border mask from warped corners (same logic as MOD_Functions)."""
        vertex = np.array(
            [[0, 0], [w, 0], [w, h], [0, h]],
            dtype=np.float32
        ).reshape(-1, 1, 2)

        homo_inv = np.linalg.inv(homo_matrix)
        vertex_trans = cv2.perspectiveTransform(vertex, homo_inv)
        vertex_transformed = np.array(vertex_trans, dtype=np.int32).reshape(1, 4, 2)

        im = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(im, vertex_transformed, True, 255)
        cv2.fillPoly(im, vertex_transformed, 255)
        mask = 255 - im

        return mask


# ══════════════════════════════════════════════════════════════
#  CPU Fallback – identical interface, uses original OpenCV code
# ══════════════════════════════════════════════════════════════

class GMC_CPU:
    """
    CPU fallback using original MOD_Functions.motion_compensate logic.
    Kept for comparison / systems without VPI.
    """

    def __init__(self, scale=2, grid_w=128, grid_h=96, outlier_dist=50.0,
                 ransac_thresh=3.0, min_points=15):
        self.scale         = scale
        self.grid_w        = grid_w
        self.grid_h        = grid_h
        self.outlier_dist  = outlier_dist
        self.ransac_thresh = ransac_thresh
        self.min_points    = min_points

    def compute_mask(self, frame_prev_gray, frame_cur_gray):
        """Same signature as GMC_VPI.compute_mask."""
        h_orig, w_orig = frame_cur_gray.shape[:2]

        lk_params = dict(
            winSize=(15, 15), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003)
        )

        proc_w = int(960 * self.scale)
        proc_h = int(540 * self.scale)
        f1 = cv2.resize(frame_prev_gray, (proc_w, proc_h), interpolation=cv2.INTER_CUBIC)
        f2 = cv2.resize(frame_cur_gray,  (proc_w, proc_h), interpolation=cv2.INTER_CUBIC)

        # Grid keypoints
        cols = int(proc_w / self.grid_w - 1)
        rows = int(proc_h / self.grid_h - 1)
        pts = []
        for i in range(cols):
            for j in range(rows):
                pts.append((np.float32(i * self.grid_w + self.grid_w / 2.0),
                            np.float32(j * self.grid_h + self.grid_h / 2.0)))

        pts_prev = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

        pts_cur, st, err = cv2.calcOpticalFlowPyrLK(f1, f2, pts_prev, None, **lk_params)

        good_new = pts_cur[st == 1]
        good_old = pts_prev[st == 1]

        motion_distance = []
        translate_x = []
        translate_y = []
        pts_new_filt = []
        pts_old_filt = []

        for (new, old) in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            d0 = np.sqrt((a - c) ** 2 + (b - d) ** 2)
            if d0 > self.outlier_dist:
                continue
            motion_distance.append(d0)
            translate_x.append(a - c)
            translate_y.append(b - d)
            pts_new_filt.append([a, b])
            pts_old_filt.append([c, d])

        pts_new_filt = np.array(pts_new_filt, dtype=np.float32)
        pts_old_filt = np.array(pts_old_filt, dtype=np.float32)

        avg_dist = float(np.mean(motion_distance)) if motion_distance else 0.0
        motion_x = float(np.mean(translate_x)) if translate_x else 0.0
        motion_y = float(np.mean(translate_y)) if translate_y else 0.0

        if len(pts_old_filt) < self.min_points:
            affine_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        else:
            affine_matrix, _ = cv2.estimateAffinePartial2D(
                pts_new_filt, pts_old_filt, maxIters=200, ransacReprojThreshold=self.ransac_thresh
            )
            if affine_matrix is None:
                affine_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

        # Create 3x3 homography matrix for interface compatibility
        homo_matrix = np.eye(3, dtype=np.float64)
        homo_matrix[0:2, :] = affine_matrix

        compensated = cv2.warpAffine(
            frame_prev_gray, affine_matrix, (w_orig, h_orig),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        )

        # Border mask
        vertex = np.array(
            [[0, 0], [w_orig, 0], [w_orig, h_orig], [0, h_orig]],
            dtype=np.float32
        ).reshape(-1, 1, 2)
        homo_inv = np.linalg.inv(homo_matrix)
        vertex_trans = cv2.perspectiveTransform(vertex, homo_inv)
        vertex_transformed = np.array(vertex_trans, dtype=np.int32).reshape(1, 4, 2)
        im = np.zeros((h_orig, w_orig), dtype=np.uint8)
        cv2.polylines(im, vertex_transformed, True, 255)
        cv2.fillPoly(im, vertex_transformed, 255)
        mask = 255 - im

        return compensated, mask, avg_dist, motion_x, motion_y, homo_matrix

    def compute_fd5_mask(self, frame1_gray, frame2_gray, frame3_gray):
        """Same signature as GMC_VPI.compute_fd5_mask."""
        f1 = cv2.GaussianBlur(frame1_gray, (11, 11), 0)
        f2 = cv2.GaussianBlur(frame2_gray, (11, 11), 0)
        f3 = cv2.GaussianBlur(frame3_gray, (11, 11), 0)

        comp1, mask1, _, _, _, _ = self.compute_mask(f1, f2)
        diff1 = cv2.absdiff(f2, comp1)

        comp2, mask2, _, _, _, _ = self.compute_mask(f3, f2)
        diff2 = cv2.absdiff(f2, comp2)

        # REPLICATE FD5_mask.py OVERFLOW BUG:
        # np.uint8 + np.uint8 overflows. (255+255=254)/2 = 127.
        motion_diff = ((diff1 + diff2) / 2).astype(np.uint8)
        return motion_diff


# ══════════════════════════════════════════════════════════════
#  Factory – auto-select best backend
# ══════════════════════════════════════════════════════════════

def create_gmc(prefer_gpu=True, silent=False, **kwargs):
    """
    Create the best available GMC engine.

    Parameters
    ----------
    prefer_gpu : bool
        If True and VPI is available, use GMC_VPI (CUDA).
        Otherwise fall back to GMC_CPU.
    silent : bool
        If True, suppress print statements.
    **kwargs
        Forwarded to the constructor.

    Returns
    -------
    GMC_VPI or GMC_CPU instance.
    """
    if prefer_gpu and VPI_AVAILABLE:
        try:
            gmc = GMC_VPI(**kwargs)
            if not silent:
                print("[GMC] Using GPU-accelerated pipeline (NVIDIA VPI / CUDA)")
            return gmc
        except Exception as e:
            if not silent:
                print(f"[GMC] VPI init failed ({e}), falling back to CPU")
    gmc = GMC_CPU(**kwargs)
    if not silent:
        print("[GMC] Using CPU pipeline (OpenCV)")
    return gmc
