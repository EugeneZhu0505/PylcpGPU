import numpy as np
try:
    import cupy as cp
    # Test if CuPy can actually be used with more comprehensive test
    try:
        test_array = cp.array([1.0, 2.0])
        cp.dot(test_array, test_array)  # Test matrix operations
        CUPY_AVAILABLE = True
    except Exception as e:
        print(f"Warning: CuPy installed but not functional ({e}), falling back to NumPy")
        import numpy as cp
        CUPY_AVAILABLE = False
except ImportError:
    import numpy as cp
    CUPY_AVAILABLE = False
    print("Warning: CuPy not available, falling back to NumPy")

import numba
import scipy.sparse as sparse
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from .integration_tools import solve_ivp_random
from .common import (ProgressBar, random_vector,
                     cart2spherical, spherical2cart)
from .governingeq import governingeq
from .obe import obe, abs2, dot, dot_and_add, cartesian_vector_tensor_dot


class obe_gpu(obe):
    """
    GPU-accelerated version of the OBE class for parallel multi-atom calculations.
    
    This class extends the original obe class to support batch processing of multiple atoms
    on GPU using CuPy for significant performance improvements.
    """
    
    def __init__(self, laserBeams, magField, hamitlonian,
                 a=np.array([0., 0., 0.]), transform_into_re_im=True,
                 use_sparse_matrices=None, include_mag_forces=True,
                 r0=np.array([0., 0., 0.]), v0=np.array([0., 0., 0.]),
                 use_gpu=True, batch_size=1000):
        """
        Initialize GPU-accelerated OBE solver.
        
        Parameters:
        -----------
        use_gpu : bool
            Whether to use GPU acceleration (requires CuPy)
        batch_size : int
            Number of atoms to process in parallel
        """
        
        # Initialize parent class
        super().__init__(laserBeams, magField, hamitlonian, a=a,
                         transform_into_re_im=transform_into_re_im,
                         use_sparse_matrices=use_sparse_matrices,
                         include_mag_forces=include_mag_forces,
                         r0=r0, v0=v0)
        
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.batch_size = batch_size
        
        if self.use_gpu:
            print(f"Using GPU acceleration with batch size: {batch_size}")
            self._convert_matrices_to_gpu()
        else:
            print("Using CPU computation")
            
        # Storage for batch results
        self.batch_sols = []
        
    def _convert_matrices_to_gpu(self):
        """Convert evolution matrices to GPU arrays."""
        if not self.use_gpu:
            return
            
        # Convert evolution matrices to CuPy arrays
        self.ev_mat_gpu = {}
        
        for key, value in self.ev_mat.items():
            if isinstance(value, dict):
                self.ev_mat_gpu[key] = {}
                for subkey, subvalue in value.items():
                    self.ev_mat_gpu[key][subkey] = self._convert_single_matrix(subvalue)
            else:
                self.ev_mat_gpu[key] = self._convert_single_matrix(value)
    
    def _convert_single_matrix(self, matrix):
        """Convert a single matrix to GPU format, handling various types."""
        import scipy.sparse as sparse
        
        if isinstance(matrix, list):
            # Handle list of matrices (e.g., from sparse conversion)
            return [cp.asarray(item.toarray() if sparse.issparse(item) else item) for item in matrix]
        elif sparse.issparse(matrix):
            # Handle sparse matrices
            return cp.asarray(matrix.toarray())
        elif isinstance(matrix, np.ndarray):
            # Handle numpy arrays, including object arrays from sparse conversion
            if matrix.dtype == object:
                # This is likely a numpy array containing sparse matrices
                if matrix.ndim == 1:
                    return [cp.asarray(item.toarray() if sparse.issparse(item) else item) for item in matrix]
                else:
                    # For higher dimensional object arrays, convert each element
                    result = np.empty_like(matrix, dtype=object)
                    flat_matrix = matrix.flatten()
                    flat_result = result.flatten()
                    for i, item in enumerate(flat_matrix):
                        flat_result[i] = cp.asarray(item.toarray() if sparse.issparse(item) else item)
                    return result.reshape(matrix.shape)
            else:
                # Regular numpy array
                return cp.asarray(matrix)
        else:
            # Fallback for other types
            return cp.asarray(matrix)
    
    def _gpu_matrix_multiply(self, matrix, vectors):
        """Perform matrix multiplication on GPU for batch of vectors."""
        if self.use_gpu:
            # Ensure both matrix and vectors are CuPy arrays
            matrix_gpu = cp.asarray(matrix)
            vectors_gpu = cp.asarray(vectors)
            return cp.dot(matrix_gpu, vectors_gpu)
        else:
            return np.dot(matrix, vectors)
    
    def __drhodt_batch(self, r_batch, t, rho_batch):
        """
        Compute drho/dt for a batch of atoms on GPU.
        
        Parameters:
        -----------
        r_batch : array_like, shape (3, N)
            Position vectors for N atoms
        t : float
            Time
        rho_batch : array_like, shape (n^2, N)
            Density matrices for N atoms (flattened)
            
        Returns:
        --------
        drhodt_batch : array_like, shape (n^2, N)
            Time derivatives of density matrices
        """
        if self.use_gpu:
            xp = cp
            ev_mat = self.ev_mat_gpu
        else:
            xp = np
            ev_mat = self.ev_mat
            
        # Initialize drho/dt with decay and Hamiltonian terms
        drhodt_batch = self._gpu_matrix_multiply(ev_mat['decay'], rho_batch)
        drhodt_batch += self._gpu_matrix_multiply(ev_mat['H0'], rho_batch)
        
        # Add laser field interactions
        for key in self.laserBeams.keys():
            if self.transform_into_re_im:
                # Compute electric field for all atoms in batch
                Eq_batch = xp.zeros((3, r_batch.shape[1]), dtype=complex)
                for i in range(r_batch.shape[1]):
                    if self.use_gpu:
                        r_cpu = cp.asnumpy(r_batch[:, i])
                        Eq = self.laserBeams[key].total_electric_field(r_cpu, t)
                        Eq_batch[:, i] = cp.asarray(Eq)
                    else:
                        Eq_batch[:, i] = self.laserBeams[key].total_electric_field(r_batch[:, i], t)
                
                for ii, q in enumerate(xp.arange(-1., 2., 1)):
                    # Real part contribution
                    field_real = (-1.)**q * xp.real(Eq_batch[2-ii, :])
                    mask = xp.abs(field_real) > 1e-10
                    if xp.any(mask):
                        contrib = self._gpu_matrix_multiply(ev_mat['reE'][key][ii], rho_batch)
                        drhodt_batch -= field_real[xp.newaxis, :] * contrib
                    
                    # Imaginary part contribution
                    field_imag = (-1.)**q * xp.imag(Eq_batch[2-ii, :])
                    mask = xp.abs(field_imag) > 1e-10
                    if xp.any(mask):
                        contrib = self._gpu_matrix_multiply(ev_mat['imE'][key][ii], rho_batch)
                        drhodt_batch -= field_imag[xp.newaxis, :] * contrib
        
        # Add magnetic field interactions
        B_batch = xp.zeros((3, r_batch.shape[1]))
        for i in range(r_batch.shape[1]):
            if self.use_gpu:
                r_cpu = cp.asnumpy(r_batch[:, i])
                B = self.magField.Field(r_cpu, t)
                B_batch[:, i] = cp.asarray(B)
            else:
                B_batch[:, i] = self.magField.Field(r_batch[:, i], t)
        
        for ii in range(3):
            if self.transform_into_re_im:
                field_vals = B_batch[ii, :]
                mask = xp.abs(field_vals) > 1e-10
                if xp.any(mask):
                    contrib = self._gpu_matrix_multiply(ev_mat['B'][ii], rho_batch)
                    drhodt_batch -= field_vals[xp.newaxis, :] * contrib
        
        return drhodt_batch
    
    def force_batch(self, r_batch, t, rho_batch):
        """
        Compute forces for a batch of atoms.
        
        Parameters:
        -----------
        r_batch : array_like, shape (3, N)
            Position vectors for N atoms
        t : float
            Time
        rho_batch : array_like, shape (n^2, N)
            Density matrices for N atoms (flattened)
            
        Returns:
        --------
        f_batch : array_like, shape (3, N)
            Force vectors for N atoms
        """
        if self.use_gpu:
            xp = cp
        else:
            xp = np
            
        # Reshape rho for observable calculation
        n = self.hamiltonian.n
        rho_reshaped = rho_batch.reshape((n, n, -1))
        
        # Ensure rho_reshaped is on GPU if using GPU
        if self.use_gpu:
            rho_reshaped = cp.asarray(rho_reshaped)
        
        f_batch = xp.zeros((3, r_batch.shape[1]))
        
        for key in self.laserBeams:
            # Compute average dipole moments
            mu_q_av = xp.zeros((3, r_batch.shape[1]), dtype=complex)
            for i in range(3):
                if self.use_gpu:
                    d_q_gpu = cp.asarray(self.hamiltonian.d_q_bare[key][i])
                    mu_q_av[i, :] = xp.trace(d_q_gpu @ rho_reshaped, axis1=0, axis2=1)
                else:
                    mu_q_av[i, :] = xp.trace(self.hamiltonian.d_q_bare[key][i] @ rho_reshaped, axis1=0, axis2=1)
            
            gamma = self.hamiltonian.blocks[self.hamiltonian.laser_keys[key]].parameters['gamma']
            
            # Compute electric field gradients for all atoms
            for i in range(r_batch.shape[1]):
                if self.use_gpu:
                    r_cpu = cp.asnumpy(r_batch[:, i])
                    delE = self.laserBeams[key].total_electric_field_gradient(r_cpu, t)
                    delE = cp.asarray(delE)
                else:
                    delE = self.laserBeams[key].total_electric_field_gradient(r_batch[:, i], t)
                
                for jj, q in enumerate(xp.arange(-1., 2., 1.)):
                    f_batch[:, i] += xp.real((-1)**q * gamma * mu_q_av[jj, i] * delE[:, 2-jj]) / 2
        
        # Add magnetic forces if enabled
        if self.include_mag_forces:
            for i in range(r_batch.shape[1]):
                if self.use_gpu:
                    r_cpu = cp.asnumpy(r_batch[:, i])
                    delB = self.magField.gradField(r_cpu)
                    
                    # Compute average magnetic moment using observable method
                    rho_single = cp.asnumpy(rho_reshaped[:, :, i])
                    av_mu = self.observable(self.hamiltonian.mu, rho_single)
                    
                    # Convert delB to numpy for cartesian_vector_tensor_dot
                    delB_cpu = cp.asnumpy(delB) if isinstance(delB, cp.ndarray) else delB
                    f_mag = cartesian_vector_tensor_dot(av_mu, delB_cpu)
                    f_batch[:, i] += cp.asarray(f_mag)
                else:
                    delB = self.magField.gradField(r_batch[:, i])
                    av_mu = self.observable(self.hamiltonian.mu, rho_reshaped[:, :, i])
                    f_mag = cartesian_vector_tensor_dot(av_mu, delB)
                    f_batch[:, i] += f_mag
        
        return f_batch
    
    def set_initial_conditions_batch(self, rho0_batch, r0_batch, v0_batch):
        """
        Set initial conditions for batch of atoms.
        
        Parameters:
        -----------
        rho0_batch : array_like, shape (n^2, N) or (n, n, N)
            Initial density matrices for N atoms
        r0_batch : array_like, shape (3, N)
            Initial positions for N atoms
        v0_batch : array_like, shape (3, N)
            Initial velocities for N atoms
        """
        n = self.hamiltonian.n
        N = r0_batch.shape[1]
        
        # Ensure rho0_batch is properly shaped
        if rho0_batch.shape == (n, n, N):
            rho0_batch = rho0_batch.reshape((n**2, N))
        elif rho0_batch.shape != (n**2, N):
            raise ValueError(f"rho0_batch should have shape ({n**2}, {N}) or ({n}, {n}, {N})")
        
        # Transform to real/imaginary representation if needed
        if self.transform_into_re_im:
            if self.use_gpu:
                U_inv_gpu = cp.asarray(self.Uinv)
                self.rho0_batch = cp.dot(U_inv_gpu, cp.asarray(rho0_batch))
            else:
                self.rho0_batch = np.dot(self.Uinv, rho0_batch)
        else:
            if self.use_gpu:
                self.rho0_batch = cp.asarray(rho0_batch)
            else:
                self.rho0_batch = rho0_batch
        
        if self.use_gpu:
            self.r0_batch = cp.asarray(r0_batch)
            self.v0_batch = cp.asarray(v0_batch)
        else:
            self.r0_batch = r0_batch
            self.v0_batch = v0_batch
    
    def evolve_motion_batch(self, t_span, freeze_axis=[False, False, False],
                           random_recoil=False, max_scatter_probability=0.1,
                           progress_bar=False, batch_info="", **kwargs):
        """
        Evolve motion for a batch of atoms in parallel.
        
        Parameters:
        -----------
        t_span : array_like
            Time span for evolution [t_start, t_end]
        freeze_axis : list of bool
            Which axes to freeze during evolution
        random_recoil : bool
            Whether to include random recoil
        max_scatter_probability : float
            Maximum scattering probability for random recoil
        progress_bar : bool
            Whether to show progress bar
        **kwargs : dict
            Additional arguments for solve_ivp
            
        Returns:
        --------
        sol_batch : list
            List of solution objects for each atom
        """
        if not hasattr(self, 'rho0_batch'):
            raise ValueError("Initial conditions not set. Call set_initial_conditions_batch first.")
        
        free_axes = np.bitwise_not(freeze_axis)
        N = self.r0_batch.shape[1]
        
        if progress_bar:
            if batch_info:
                print(f"Evolving {batch_info} - {N} atoms...")
            progress = progressBar()
            print(f"Time evolution from {t_span[0]:.3f}s to {t_span[1]:.3f}s")
        
        def dydt_batch(t, y_batch):
            """Batch version of the evolution equations."""
            if progress_bar:
                # Show progress based on time evolution
                time_progress = (t - t_span[0]) / (t_span[1] - t_span[0])
                progress.update(time_progress)
            
            # Reshape y_batch: [rho (n^2 x N), v (3 x N), r (3 x N)]
            n_sq = self.hamiltonian.n**2
            rho_batch = y_batch[:n_sq*N].reshape((n_sq, N))
            v_batch = y_batch[n_sq*N:n_sq*N+3*N].reshape((3, N))
            r_batch = y_batch[n_sq*N+3*N:].reshape((3, N))
            
            # Compute derivatives
            drhodt_batch = self.__drhodt_batch(r_batch, t, rho_batch)
            F_batch = self.force_batch(r_batch, t, rho_batch)
            
            if self.use_gpu:
                # Convert to numpy for scipy integration
                drhodt_batch = cp.asnumpy(drhodt_batch)
                F_batch = cp.asnumpy(F_batch)
                free_axes_gpu = cp.asarray(free_axes)
                constant_accel_gpu = cp.asarray(self.constant_accel)
                
                dvdt_batch = (F_batch * free_axes_gpu[:, np.newaxis] / self.hamiltonian.mass + 
                             constant_accel_gpu[:, np.newaxis])
            else:
                dvdt_batch = (F_batch * free_axes[:, np.newaxis] / self.hamiltonian.mass + 
                             self.constant_accel[:, np.newaxis])
            
            drdt_batch = v_batch
            
            return np.concatenate([
                drhodt_batch.flatten(),
                dvdt_batch.flatten(),
                drdt_batch.flatten()
            ])
        
        # Prepare initial conditions
        if self.use_gpu:
            y0 = np.concatenate([
                cp.asnumpy(self.rho0_batch).flatten(),
                cp.asnumpy(self.v0_batch).flatten(),
                cp.asnumpy(self.r0_batch).flatten()
            ])
        else:
            y0 = np.concatenate([
                self.rho0_batch.flatten(),
                self.v0_batch.flatten(),
                self.r0_batch.flatten()
            ])
        
        # Solve the system
        if not random_recoil:
            sol = solve_ivp(dydt_batch, t_span, y0, **kwargs)
        else:
            # For random recoil, we need to implement batch version
            # This is more complex and would require custom implementation
            raise NotImplementedError("Random recoil not yet implemented for batch processing")
        
        if progress_bar:
            progress.update(1.)
        
        # Reshape and store results
        self._reshape_batch_sol(sol, N)
        
        return self.sol_batch
    
    def _reshape_batch_sol(self, sol, N):
        """
        Reshape batch solution into individual atom solutions.
        """
        n_sq = self.hamiltonian.n**2
        n = self.hamiltonian.n
        
        self.sol_batch = []
        
        for i in range(N):
            # Create individual solution object
            class AtomSolution:
                pass
            
            atom_sol = AtomSolution()
            atom_sol.t = sol.t
            
            # Extract data for this atom
            rho_flat = sol.y[:n_sq, :]
            v_data = sol.y[n_sq:n_sq+3*N, :].reshape((3, N, -1))
            r_data = sol.y[n_sq+3*N:, :].reshape((3, N, -1))
            
            # Reshape rho back to matrix form
            if self.transform_into_re_im:
                rho_complex = np.zeros((n_sq, len(sol.t)), dtype=complex)
                for j in range(len(sol.t)):
                    rho_complex[:, j] = self.U @ rho_flat[:, j]
                atom_sol.rho = rho_complex.reshape((n, n, -1))
            else:
                atom_sol.rho = rho_flat.reshape((n, n, -1))
            
            atom_sol.v = v_data[:, i, :]
            atom_sol.r = r_data[:, i, :]
            
            self.sol_batch.append(atom_sol)
    
    def observable_batch(self, O, rho_batch=None):
        """
        Compute observable for batch of atoms.
        
        Parameters:
        -----------
        O : array_like, shape (n, n)
            Observable operator
        rho_batch : array_like, shape (n, n, N, T), optional
            Density matrices for N atoms at T time points
            
        Returns:
        --------
        obs_batch : array_like, shape (N, T)
            Observable values for each atom at each time point
        """
        if rho_batch is None:
            if not hasattr(self, 'sol_batch'):
                raise ValueError("No solution available. Run evolve_motion_batch first.")
            
            # Extract rho from solutions
            N = len(self.sol_batch)
            T = len(self.sol_batch[0].t)
            n = self.hamiltonian.n
            
            rho_batch = np.zeros((n, n, N, T), dtype=complex)
            for i, sol in enumerate(self.sol_batch):
                rho_batch[:, :, i, :] = sol.rho
        
        if self.use_gpu:
            O_gpu = cp.asarray(O)
            rho_gpu = cp.asarray(rho_batch)
            
            # Compute trace(O @ rho) for each atom and time point
            obs_batch = cp.trace(O_gpu @ rho_gpu, axis1=0, axis2=1)
            
            if cp.allclose(cp.imag(obs_batch), 0):
                return cp.asnumpy(cp.real(obs_batch))
            else:
                return cp.asnumpy(obs_batch)
        else:
            obs_batch = np.trace(O @ rho_batch, axis1=0, axis2=1)
            
            if np.allclose(np.imag(obs_batch), 0):
                return np.real(obs_batch)
            else:
                return obs_batch