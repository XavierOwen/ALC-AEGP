import numpy as np
import torch
import scipy
from scipy.stats import qmc
from smt.sampling_methods import LHS

def generate_dataset_for_borehole(
    N_train=10,
    N_test=500,
    N_ref=100,
    N_cand=100,
    noise_scale=0.1,
    device="cpu",
    seed=None
):
    """
    Generate active learning sets using the BH groundwater flow model.

    Inputs are normalized to [0, 1]^8 using Latin Hypercube Sampling.

    Returns:
        x_train, y_train
        x_test, y_test
        x_ref, y_ref
        x_cand, y_cand
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    def f(x_zeroOne):
        # x_zeroOne: numpy array of shape (N, 8) in [0,1]^8
        l_bounds = np.array([0.05, 100, 63070, 990, 63.1, 700, 1120, 9855])
        u_bounds = np.array([0.15, 50000, 115600, 1110, 116, 820, 1680, 12045])
        x_physical = l_bounds + x_zeroOne * (u_bounds - l_bounds)

        r_w = x_physical[:, 0]
        r   = x_physical[:, 1]
        T_u = x_physical[:, 2]
        H_u = x_physical[:, 3]
        T_l = x_physical[:, 4]
        H_l = x_physical[:, 5]
        L   = x_physical[:, 6]
        K_w = x_physical[:, 7]

        numerator = 2 * np.pi * T_u * (H_u - H_l)
        log_rrw = np.log(r / r_w)
        denominator = log_rrw * (1 + 2 * L * T_u / (log_rrw * r_w**2 * K_w) + T_u / T_l)
        y = numerator / denominator
        return y.reshape(-1, 1).astype(np.float32)

    def sample_lhs(N):
        sampler = qmc.LatinHypercube(d=8, seed=seed)
        return sampler.random(n=N)  # shape (N, 8) in [0,1]

    def add_noise(y, scale):
        return y + np.random.randn(*y.shape) * scale

    def to_tensor(x, y):
        return (
            torch.tensor(x, dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.float32, device=device),
        )

    # Training
    x_train_np = sample_lhs(N_train)
    y_train_np = f(x_train_np)
    y_train_np = add_noise(y_train_np, noise_scale)

    # Test
    x_test_np = sample_lhs(N_test)
    y_test_np = f(x_test_np)
    y_test_np = add_noise(y_test_np, noise_scale)

    # Reference
    x_ref_np = sample_lhs(N_ref)
    y_ref_np = f(x_ref_np)

    # Candidate
    x_cand_np = sample_lhs(N_cand)
    y_cand_np = f(x_cand_np)
    y_cand_np = add_noise(y_cand_np, noise_scale)

    return (
        *to_tensor(x_train_np, y_train_np),
        *to_tensor(x_test_np, y_test_np),
        *to_tensor(x_ref_np, y_ref_np),
        *to_tensor(x_cand_np, y_cand_np)
    )

def generate_dataset_for_3d(
    N_train=10,
    N_test=500,
    N_ref=100,
    N_cand=100,
    noise_scale=0.1,
    device="cpu",
    seed=None
):
    """
    Generate active learning sets on a unit sphere with v=cos(phi), theta parameterization.
    Target function: f(x) = cos(x1) + x2^2 + exp(x3)

    Returns:
        x_train, y_train
        x_test, y_test
        x_ref, y_ref
        x_cand, y_cand
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    def f(x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        return (np.cos(x1) + x2**2 + np.exp(x3)).reshape(-1, 1).astype(np.float32)

    def sample_sphere(N):
        # v in [-1,1], theta in [0, 2pi]
        xlimits = np.array([[-1, 1], [0, 2*np.pi]])
        lhs = LHS(xlimits=xlimits, criterion="maximin")
        v_theta = lhs(N)
        v = v_theta[:, 0]
        theta = v_theta[:, 1]

        x = np.zeros((N, 3), dtype=np.float32)
        sin_phi = np.sqrt(1 - v**2)
        x[:, 0] = sin_phi * np.cos(theta)
        x[:, 1] = sin_phi * np.sin(theta)
        x[:, 2] = v

        y = f(x)
        return x, y

    def add_noise(y, scale):
        return y + np.random.randn(*y.shape) * scale

    def to_tensor(x, y):
        return (
            torch.tensor(x, dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.float32, device=device),
        )

    # Training
    x_train_np, y_train_np = sample_sphere(N_train)
    y_train_np = add_noise(y_train_np, noise_scale)

    # Test
    x_test_np, y_test_np = sample_sphere(N_test)
    y_test_np = add_noise(y_test_np, noise_scale)

    # Reference
    x_ref_np, y_ref_np = sample_sphere(N_ref)

    # Candidate
    x_cand_np, y_cand_np = sample_sphere(N_cand)
    y_cand_np = add_noise(y_cand_np, noise_scale)

    return (
        *to_tensor(x_train_np, y_train_np),
        *to_tensor(x_test_np, y_test_np),
        *to_tensor(x_ref_np, y_ref_np),
        *to_tensor(x_cand_np, y_cand_np)
    )

def generate_dataset_for_2d(
    N_train=10,
    N_test=500,
    N_ref=100,
    N_cand=100,
    bounds=[0., 10.0],
    degree=45,
    noise_scale=0.1,
    device="cpu",
    seed=None
):
    """
    Generate active learning sets using rotated 2D Gaussian mixture function.

    Returns:
        x_train, y_train
        x_test, y_test
        x_ref, y_ref
        x_cand, y_cand
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    def f(x_np):  # expects numpy array, returns numpy array
        phi = degree * np.pi / 180
        rot = np.array([
            [np.cos(phi), -np.sin(phi)],
            [np.sin(phi),  np.cos(phi)]
        ])
        x_rot = x_np @ rot.T
        y1 = scipy.stats.norm(3, 0.5).logpdf(x_rot[:, [1]])
        y2 = scipy.stats.norm(-3, 0.5).logpdf(x_rot[:, [1]])
        y = 1 - np.exp(y1) - np.exp(y2) + x_rot[:, [0]] / 100
        return y.astype(np.float32)

    def lhs_sample(N, bounds, d=2):
        bounds_np = np.array([bounds] * d)
        lhs = LHS(xlimits=bounds_np, criterion="maximin")
        return lhs(N)

    # Training set
    x_train_np = lhs_sample(N_train, bounds)
    y_train_np = f(x_train_np) + np.random.randn(N_train, 1) * noise_scale

    # Test set
    x_test_np = lhs_sample(N_test, bounds)
    y_test_np = f(x_test_np) + np.random.randn(N_test, 1) * noise_scale

    # Reference set
    x_ref_np = lhs_sample(N_ref, bounds)
    y_ref_np = f(x_ref_np)

    # Candidate set
    x_cand_np = lhs_sample(N_cand, bounds)
    y_cand_np = f(x_cand_np) + np.random.randn(N_cand, 1) * noise_scale

    # Convert to torch
    def to_tensor(x, y):
        return (
            torch.tensor(x, dtype=torch.float, device=device),
            torch.tensor(y, dtype=torch.float, device=device)
        )

    return (
        *to_tensor(x_train_np, y_train_np),
        *to_tensor(x_test_np, y_test_np),
        *to_tensor(x_ref_np, y_ref_np),
        *to_tensor(x_cand_np, y_cand_np)
    )

def generate_dataset_for_1d(
    N_train=10,
    N_test=500,
    N_ref=100,
    N_cand=100,
    bounds = [0.0, 1.0],
    noise_scale=0.1,
    device="cpu",
    seed=None,
):
    """
    Generate all necessary sets for active learning GP on Heaviside function.
    
    Returns:
        x_train, y_train
        x_test, y_test
        x_ref, y_ref
        x_cand, y_cand
        y_mean, y_std (used if standardization applied)
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    def lhs_sample(N, bounds):
        lhs = LHS(xlimits=np.array([bounds]))
        lhs = np.sort(lhs(N),axis=0)
        return torch.tensor(lhs, dtype=torch.float64, device=device)
    def f(x):
        """
        Vectorized piecewise cosine function.
        Args:
            x (torch.Tensor): Tensor of shape (N, 1) with values in [0, 1]
        Returns:
            torch.Tensor: Same shape as x
        """
        y = torch.full_like(x, 1.35)
        y[x <= 0.33] *= torch.cos(12 * torch.pi * x[x <= 0.33])
        y[x >= 0.66] *= torch.cos(6 * torch.pi * x[x >= 0.66])
        return y

    # --- 1. Training set ---
    x_train = lhs_sample(N_train, bounds)
    y_clean_train = f(x_train)
    y_train = y_clean_train + torch.randn_like(y_clean_train) * noise_scale

    # --- 2. Test set (LHS in [-5, 5]) ---
    x_test = lhs_sample(N_test, bounds)
    y_clean_test = f(x_test)
    y_test = y_clean_test + torch.randn_like(y_clean_test) * noise_scale

    # --- 3. Reference set (LHS in [-2, 2]) ---
    x_ref = lhs_sample(N_ref, [0.0, 1.0])
    y_ref = f(x_ref) #

    # --- 4. Candidate set (LHS in [-2, 2]) ---
    x_cand = lhs_sample(N_cand, [0.0, 1.0])
    y_clean_cand = f(x_cand)
    y_cand = y_clean_cand + torch.randn_like(y_clean_cand) * noise_scale

    return x_train, y_train, x_test, y_test, x_ref, y_ref, x_cand, y_cand