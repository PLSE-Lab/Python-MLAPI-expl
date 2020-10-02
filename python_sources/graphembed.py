from numpy import square, sqrt
from scipy.integrate import quad
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from multiprocessing import Pool, cpu_count
from numpy import copy
from scipy import sparse


class LanczosPolynomial:
    """
    Container for Lanczos polynomial:
        Sequence of polynomials of increasing order (by one) that are mutually orthogonal 
        as per some inner product space. They can be evaluated via a 3-term recursion and 
        this class holds the polynomial and associated evaluation methods. They span all 
        polynomials upto the specified order
    """

    def __init__(self, phi, order):
        """
        Builds up the Lanczos polynomial series up to the requested order
        :param phi: function that defines the inner product space 
                             (needs to be non-negative in [-1, 1])
        :param order: max order of lanczos series
        :return: none
        """
        if order < 0:
            ValueError("order must be no smaller than 0")

        self.phi = phi

        # initialize Lanczos series with zeroth order unit-norm polynomial
        result = quad(self.phi, -1, 1)
        beta_squared = result[0]
        self.alpha = []
        self.beta = [sqrt(beta_squared)]
        self.r = 0

        # recurse up to specified order
        for i in range(1, order + 1):
            self.add_order()

    @staticmethod
    def _evaluate_poly(x, alpha, beta, r):
        """
        Computes the r-th order Lanczos polynomial defined by the
        3-recursion from lists alpha and beta
        :param x: position where we wish to evaluate the Lanczos polynomial
        :param alpha: defines the Lanczos recursion
        :param beta: defines the Lanczos recursion
        :param r: order of the polynomial; r <= len(alpha)
        :return: evaluate r-th order Lanczos polynomial @ x
        """

        parent = 1. / beta[0]
        grand_parent = 0.

        def eval_current(i, x_i, x_i_minus_1):
            return ((x - alpha[i - 1]) * x_i - beta[i - 1] * x_i_minus_1) / beta[i]

        # recurse to r-th order polynomial
        for j in range(1, r + 1):
            current = eval_current(j, parent, grand_parent)
            grand_parent = parent
            parent = current

        return parent

    def evaluate_poly(self, x, r=None):
        """
        Calls the evaluation of the r-th order Lanczos polynomial defined by
        the 3-recursion from lists alpha and beta at x
        :param x: position where this evaluation ought to be performed
        :param r: order of the Lanczos series (Defaults to the highest order available)
        :return: value of r-th polynomial at x
        """
        if r is None:
            r = self.r
        elif r is -1:
            return 0.
        elif r > self.r:
            raise ValueError("Supply order less than or equal to class instance's order {0}".
                             format(self.r))

        return self._evaluate_poly(x, self.alpha, self.beta, r)

    def inner_product(self, fun, r):
        """
        Evaluate inner product of the function fun with the r-th order Lanczos polynomial
         corresponding to self.phi
        :param fun: function
        :param r: order of Lanczos polynomial
        :return: inner product value
        """
        if r > self.r:
            ValueError("Requested order {0} exceeds polynomials that have "
                       "been generated {1}".format(r, self.r))

        result = quad(lambda x: fun(x) * self.evaluate_poly(x, r) * self.phi(x), -1, 1, limit=100)
        inner_product = result[0]

        return inner_product

    def add_order(self):
        """
        Increase order of Lanczos approximation
        :return: None
        """

        # subtracting beta[r-1] times r-1-th order lanczos polynomial ensures
        # that  x * r-th order Lanczos polynomial is orthogonal to r-1-th
        # order lanczos polynomial
        def next_lanczos_poly(x):
            return x * self.evaluate_poly(x, self.r) - self.beta[-1] * self.evaluate_poly(x, self.r - 1)

        # compute inner product with r-th order lanczos polynomial
        alpha_new = self.inner_product(next_lanczos_poly, self.r)

        # remove projection along the r-th order lanczos polynomial
        def next_lanczos_poly_orth(x):
            return (x - alpha_new) * self.evaluate_poly(x, self.r) - self.beta[-1] * self.evaluate_poly(x, self.r - 1)

        # compute beta_new = norm as per ip space of next_lanczos_poly_orth
        result = quad(lambda x: self.phi(x) * square(next_lanczos_poly_orth(x)), -1, 1, limit=100)
        beta_new_squared = result[0]
        beta_new = sqrt(beta_new_squared)

        # update container
        self.alpha.append(alpha_new)
        self.beta.append(beta_new)
        self.r += 1

    def order(self):
        return self.r


class PolynomialApproximator:
    """
    Container for polynomial approximation of a function
    """

    def __init__(self, fun, phi, order):
        """
        Best r-th order polynomial approximation of fun as measured by:
            integral in -1 to 1 of ( phi(x) (fun(x) - polynomial_approximation(x))^2 )
        :param fun: function for which we seek a polynomial approximation
        :param phi: function that defines the inner product space (needs to be positive in [-1, 1])
        :param order: order of the polynomial approximation
        :return: None
        """
        self.fun = fun
        self.phi = phi

        self.lanczos_series = None
        self.order = None
        self.lanczos_coeff = None
        result = quad(lambda x: self.phi(x) * square(fun(x)), -1, 1, limit=100)
        squared_norm = result[0]
        self.approx_error = squared_norm

        self._perform_approximation(order)

    def _perform_approximation(self, order):
        """
        Driver method that obtains the projection of fun onto a Lanczos polynomial series
        and populates the order, lanczos_series, approx_error and lanczos_coeff fields
        """
        # zeroth order approximation first
        self.lanczos_series = LanczosPolynomial(self.phi, 0)
        self.order = 0
        new_coeff = self.lanczos_series.inner_product(self.fun, 0)
        self.lanczos_coeff = [new_coeff]
        self.approx_error -= new_coeff * new_coeff

        # build up to requested order
        for r in range(1, order + 1):
            self.add_order()

    def approx_details(self):
        """
        :return: information needed to write down the polynomial approximation
        """
        return self.lanczos_series.alpha, self.lanczos_series.beta, self.lanczos_coeff

    def _eval_approx(self, x, r):
        """
        evaluates the polynomial approximation of order r @ x
        """
        current = 1. / self.lanczos_series.beta[0]

        grand_parent = 0.
        parent = current
        cum_eval = self.lanczos_coeff[0] * current

        def eval_current(i, x_i, x_i_minus_1):
            return ((x - self.lanczos_series.alpha[i - 1]) * x_i - self.lanczos_series.beta[i - 1] * x_i_minus_1) \
                   / self.lanczos_series.beta[i]

        # recurse to r-th order polynomial
        for j in range(1, r + 1):
            current = eval_current(j, parent, grand_parent)
            grand_parent = parent
            parent = current
            cum_eval += self.lanczos_coeff[j] * current

        return cum_eval

    def eval_approx(self, x, r=None):
        """
        evaluates the polynomial approximation of order r @ x
        :param x: point where we want the approximation evaluated
        :param r: order of the polynomial approximation (defaults to the largest order available)
        """
        if r is None:
            r = self.order

        if r < 0:
            raise ValueError("order must be positive")
        elif r > self.order:
            raise ValueError("order cannot be greater than the polynomial approximation order")
        elif r > len(self.lanczos_coeff) - 1:
            raise ValueError("This approximation is not yet available")

        return self._eval_approx(x, r)

    def add_order(self):
        """
        Improve the polynomial approximation by increasing its order
        """
        # get the next order in the lanczos polynomial series
        self.lanczos_series.add_order()

        # project out what has been approximated so far
        # though this is not needed (we can self.fun instead of residual when we compute new_coeff below)
        # we leave this in for stability
        def residual(x):
            return self.fun(x) - self._eval_approx(x, self.order)

        # compute inner product of the residual with the newly extracted (self.order + 1)-th \
        # order Lanczos polynomial
        new_coeff = self.lanczos_series.inner_product(residual, self.order + 1)

        # update polynomial approximation using the inner product
        self.lanczos_coeff.append(new_coeff)
        self.order += 1
        self.approx_error -= new_coeff * new_coeff


def spectral_norm(args):
    """
    runs power iteration on the columns of vecs to estimate the spectral norm of a symmetric matrix
    :param args: (mat, vecs, iter_max)
            mat: a square sparse scipy matrix
            vecs: A numpy matrix of the same length as A
            iter_max: number of iterations
    :return: An array of lower bounds of the spectral norm of the symmetric matrix mat
    """
    # unpack arguments
    mat, vecs, iter_max = args
    del args

    if mat.shape[0] != mat.shape[1]:
        raise ValueError('Matrix must be square')

    sigma_max_est = 0

    for iter_count in range(iter_max):
        vecs_new = mat * vecs
        vecs_new_norm = np.linalg.norm(vecs_new, axis=0)

        if iter_count == (iter_max - 1):
            vecs_norm = np.linalg.norm(vecs, axis=0)
            sigma_max_est = [y / x for (x, y) in zip(vecs_norm, vecs_new_norm)]

        vecs = vecs_new * csc_matrix(([1.0 / x for x in vecs_new_norm], (range(vecs.shape[1]), range(vecs.shape[1]))))

    return sigma_max_est


def compute_embed(args):
    """
    Worker function which computes the embedding
    :param args: tuple (mat, mat_spec_norm, proj_mat, boost, lanczos_coeff, alpha, beta, verbose)
    :return: embedding matrix of same size as as embed_mat
    """

    mat, mat_spec_norm, proj_mat, boost, lanczos_coeff, alpha, beta, verbose = args

    del args

    if not isinstance(mat, csr_matrix):
        raise ValueError('Matrix must be a sparse CSR matrix')

    if mat.shape[0] != mat.shape[1]:
        raise ValueError('Matrix must be square')

    if mat.shape[1] != proj_mat.shape[0]:
        raise ValueError('Projection vectors must match matrix size')

    if boost <= 0:
        raise ValueError('Boost parameter must be a positive integer')

    # core lanczos iteration
    def eval_current(i, x_i, x_i_minus_1):
        # return (mat*x_i/mat_spec_norm - alpha[i-1]*x_i - beta[i-1]*x_i_minus_1)/beta[i]
        a = (1. / (mat_spec_norm * beta[i])) * (mat * x_i)
        b = (beta[i - 1] / beta[i]) * x_i_minus_1
        if alpha[i - 1] == 0:
            return a - b
        else:
            return a - b - (alpha[i - 1] / beta[i]) * x_i

    embedding = proj_mat

    for boost_idx in range(boost):
        current_vectors = embedding / beta[0]
        # zeroth order polynomial approximation of embedding function
        embedding = lanczos_coeff[0] * current_vectors

        if verbose:
            ndims = proj_mat.shape[1]
            energy_lanczos = np.sum(np.square(current_vectors)) / ndims
            energy_embedding = np.sum(np.square(embedding)) / ndims
            print ("Boost {}; Lanczos {}; Energy Lanczos {}; Energy embedding {}".format(
                boost_idx + 1, 0, energy_lanczos, energy_embedding))

        parent_vectors = current_vectors
        grand_parent_vectors = np.zeros(current_vectors.shape)

        # recurse to r-th order polynomial approximation of
        # spectral embedding
        for r in range(1, len(lanczos_coeff)):
            current_vectors = eval_current(r, parent_vectors, grand_parent_vectors)
            embedding += lanczos_coeff[r] * current_vectors

            if verbose:
                ndims = proj_mat.shape[1]
                energy_lanczos = np.sum(np.square(current_vectors)) / ndims
                energy_embedding = np.sum(np.square(embedding)) / ndims
                print ("Boost {}; Lanczos {}; Energy Lanczos {}; Energy embedding {}".format(
                    boost_idx + 1, r, energy_lanczos, energy_embedding))

            grand_parent_vectors = parent_vectors
            parent_vectors = current_vectors

    return embedding


class FastEmbed:
    """
    Class that contains the compressed embedding obtained using the FastEmbed algorithm and the associated parameters
    """
    BOOST = 2
    POLY_ORDER = 60

    @staticmethod
    def _default_phi(_):
        return 1.

    SCALE_UP_SPEC_NORM_EST = 1.01
    SPEC_NORM_EST_ITER = 80
    SPEC_NORM_EST_DIMS = 20
    NORMALIZE_FUN = True
    N_JOBS = -1
    VERBOSE = False
    CONFIG_INPUT = {'spec_norm_est_dims', 'spec_norm_est_iter', 'poly_order', 'scale_up_spec_norm_est', 'n_jobs',
                    'sigma_max', 'boost', 'normalize_fun', 'verbose'}

    def __init__(self, mat, fun, embed_dims, config=None, phi=None):
        """
        Embedding object for symmetric matrices
        :param mat: A square symmetric sparse scipy matrix in CSR format
        :param fun: Spectral weighting function used to weight eigenvectors
        :param embed_dims: Number of dims for the embedding
        :param phi: Approximation error weighting function
        :param config: (Optional) Dict with algorithm configuration parameters
        :return: FastEmbed object
        """
        # we need csr matrix for mat-vec multiplications
        if not isinstance(mat, csr_matrix):
            raise ValueError("Input matrix must be a square symmetric scipy "
                             "sparse symmetric matrix")

        self.mat = mat
        self.n = mat.shape[0]
        self.fun = fun
        self.normalize_fun = FastEmbed.NORMALIZE_FUN
        self.embed_dims = embed_dims
        if phi is None:
            self.phi = FastEmbed._default_phi
        else:
            self.phi = phi

        self.poly_order = FastEmbed.POLY_ORDER
        self.boost = FastEmbed.BOOST

        self.spec_norm_est_dims = FastEmbed.SPEC_NORM_EST_DIMS
        self.spec_norm_est_iter = FastEmbed.SPEC_NORM_EST_ITER
        self.scale_up_spec_norm_est = FastEmbed.SCALE_UP_SPEC_NORM_EST
        self.n_jobs = FastEmbed.N_JOBS
        self.sigma_max = None
        self.verbose = FastEmbed.VERBOSE

        if config is not None:
            self._load_config(config)

        if self.n_jobs == -1:
            self.n_jobs = cpu_count()

        if self.sigma_max is None:
            self.sigma_max = self._est_sigma_max()

        self.poly_approx = self._approx_poly(fun, self.boost, self.sigma_max, self.normalize_fun, self.phi,
                                             self.poly_order)

        if self.verbose:
            print ("{0} concurrent jobs will be used".format(self.n_jobs))
            print ("Spectral norm of matrix is {0}".format(self.sigma_max))
            print ("Polynomial approximation error is {0}".format(self.poly_approx.approx_error))
            print ("Lanczos polynomial series recursion\nalpha {}\nbeta {}".format(
                self.poly_approx.lanczos_series.alpha,
                self.poly_approx.lanczos_series.beta,
            ))
            print ("Lanczos mixing weights {}".format(self.poly_approx.lanczos_coeff))
            print ("Proceeding to compute embedding ...")

        self.proj_mat = (2. * np.random.randint(2, size=(self.n, self.embed_dims)) - 1.) / \
                        float(np.sqrt(self.embed_dims))
        self.embed = None
        # Fill up self.embed
        self._eval_embed()

    def _load_config(self, config):
        """
        Loads parameters specified via the config dict
        """

        for key in FastEmbed.CONFIG_INPUT:
            if key in config:
                setattr(self, key, config[key])

    @staticmethod
    def _approx_poly(fun, boost, sigma_max, normalize_fun, phi, poly_order):
        """
        Sets up the polynomial approximation of the spectral weighting function fun using the relevant parameters
        """
        if normalize_fun:
            if boost is 1:
                return PolynomialApproximator(fun, phi, poly_order)
            elif boost > 1:
                def fun_prime(x):
                    return np.power(np.abs(fun(x)), 1. / boost) * np.sign(fun(x))

                return PolynomialApproximator(fun_prime, phi, poly_order)
            else:
                raise ValueError('Boost parameter cannot be smaller than 1')

        if boost is 1:
            def fun_prime(x):
                return fun(1. * x / sigma_max)
            PolynomialApproximator(fun_prime, phi, poly_order)
        elif boost > 1:
            def fun_prime(x):
                y = 1. * x / sigma_max
                return np.power(np.abs(fun(y)), 1. / boost) * np.sign(fun(y))
            PolynomialApproximator(fun_prime, phi, poly_order)
        else:
            raise ValueError('Boost parameter cannot be smaller than 1')

    def _est_sigma_max(self):
        """
            Sets up a parallel implementation of spectral norm estimation via Power Iteration
            using Python's multiprocessing library (Driver method)
        """

        start_vecs = 2 * np.random.randint(2, size=(self.n, self.spec_norm_est_dims)) - 1

        n_jobs = self.n_jobs
        n_jobs = min(n_jobs, self.spec_norm_est_dims)

        if n_jobs == 1:
            # no need the over head of a new process
            sigma_max_vecs = spectral_norm((self.mat, start_vecs, self.spec_norm_est_iter))
            sigma_max = self.scale_up_spec_norm_est * float(max(sigma_max_vecs))
            del start_vecs
            return sigma_max

        pool = Pool(processes=n_jobs)

        # slice projection matrix (pick columns)
        each_dim = self.spec_norm_est_dims // n_jobs
        indices = [[k for k in range(r * each_dim, min((r + 1) * each_dim, self.spec_norm_est_dims))] for r in range(n_jobs)]
        indices[-1].extend([k for k in range(each_dim * n_jobs, self.spec_norm_est_dims)])

        # different arguments for each processor
        args_list = [(self.mat, start_vecs[:, indices[r]], self.spec_norm_est_iter) for r in range(n_jobs)]

        # map followed by reduce step (concatenate results together)
        sigma_max_vecs = np.concatenate(pool.map(spectral_norm, args_list), axis=0)

        # kill processes
        pool.terminate()

        return self.scale_up_spec_norm_est * float(max(sigma_max_vecs))

    def _eval_embed(self):
        """
        Sets up a parallel implementation of FastEmbed using python's multiprocessing library
        """
        n_jobs = self.n_jobs
        n_jobs = min(n_jobs, self.embed_dims)

        if n_jobs == 1:
            # no need the over head of a new process
            self.embed = compute_embed((self.mat, self.sigma_max, np.copy(self.proj_mat), self.boost,
                                        self.poly_approx.lanczos_coeff, self.poly_approx.lanczos_series.alpha,
                                        self.poly_approx.lanczos_series.beta, self.verbose))
            return

        # create n_jobs processors
        pool = Pool(processes=n_jobs)

        # slice projection matrix (pick columns)
        each_dim = self.embed_dims // n_jobs
        indices = [[k for k in range(r * each_dim, min((r + 1) * each_dim, self.embed_dims))]
                   for r in range(n_jobs)]
        indices[-1].extend([k for k in range(each_dim * n_jobs, self.embed_dims)])

        # different arguments for each processor

        args = [(self.mat, self.sigma_max, self.proj_mat[:, indices[r]], self.boost,
                 self.poly_approx.lanczos_coeff, self.poly_approx.lanczos_series.alpha,
                 self.poly_approx.lanczos_series.beta, self.verbose) for r in range(n_jobs)]

        self.embed = np.concatenate(pool.map(compute_embed, args), axis=1)


def fast_embed_eig(mat, fun, embed_dims, config=None, phi=None):
    """
    Wrapper for computing eigenvector embeddings a symmetric matrix using the FastEmbed class
    :param mat: symmetric matrix for which we seek an eigenvector embedding
    :param fun: How to weight eigenvectors relative to their eigenvalue / maximum abs(eigenvalue)
    :param embed_dims: size of embedding
    :param config: ***** Optional **** configuration parameters - a dict can take keys in:
        'poly_order' - What order of the polynomial approximation approximation of the embedding function 'fun' is
                                needed? Depends on how sharp the function 'fun' is and the size of the matrix. For high
                                dimensional matrices "errors" in approximation can add up. So larger values of 'poly_order'
                                are recommended
        'boost' - How to approximate 'fun'? If nulls in the spectrum are important (which is typically the case when we
                                want to suppress noise), we recommend setting boost to 2 or 3 (as opposed to 1). In this
                                case, we obtain the polynomial approximation of fun(x) ^ (1/boost) and cascade the
                                 algorithm 'boost' number of times. See the paper in the README for details
        'sigma_max' - The spectral norm of the matrix 'mat' computed elsewhere. If this parameter is specified we do not
                                attempt to estimate the spectral norm of the matrix and hence the parameters
                                'spec_norm_est_dims', 'spec_norm_est_iter' and 'scale_up_spec_norm_est' are unused
        'spec_norm_est_dims' - Number of random starting vectors to use to estimate the spectral norm of the matrix
                                via power iteration
        'spec_norm_est_iter' - Number of iterates of power iteration for estimating the spectral norm (depends on how
                                clustered the eigen values are around the spectral norm)
        'scale_up_spec_norm_est' - After estimating the spectral norm using power iteration (this is strictly a lower
                                bound on the spectral norm), we multiply it by a small number greater than 1 given by
                                scale_up_spec_norm_est (defaults to FastEmbed.SCALE_UP_SPEC_NORM_EST) to potentially
                                arrive at an upper bound. Since the algorithm assumes that we have an upper bound on the
                                spectral norm of the matrix, this parameter alongside 'spec_norm_est_dims',
                                'spec_norm_est_iter' may have to be tuned if the algorithm diverges.
                                Another option in those cases is to provide the spectral norm computed elsewhere
                                using the parameter sigma_max
        'n_jobs' - Number of cores to use in embedding computation (defaults to -1, which denotes all available cores)
        'normalize_fun' - if set to True (default), fun represents the scaling of the eigen value relative to the
                                spectral norm (largest magnitude eigenvalue) ie,
                                f(eigenvalue/spectralnorm) * eigenvector; if set to False, we use fun as is on the
                                eigenvalue to obtain its weight ie., f(eigenvalue) * eigenvector
         'verbose' - if set to True - prints details along the way; Defaults to False
    :param phi **** Optional ****
        how to weight polynomial approximation errors
        integrate(phi(x) * (fun(x) - poly_approx_of_fun(x))^2, -1, 1) (Defaults to phi(x) = 1.)
    :return: Embedding of rows/cols of size (#rows, embed_dims) of mat
    """
    embed = FastEmbed(sparse.csr_matrix(mat), fun, embed_dims, config, phi)
    embed_rows_cols = copy(embed.embed)
    return embed_rows_cols


def fast_embed_svd(mat, fun, embed_dims, config=None, phi=None):
    """
    Wrapper for computing SVD embeddings using the FastEmbed class
    :param mat: matrix for which we seek an SVD embedding
    :param fun: How to weight singular vectors relative to their singular value / maximum singular value
    :param embed_dims: Size of embedding
    :param config: ***** Optional **** configuration parameters - a dict can take keys in:
        'poly_order' - What order of the polynomial approximation approximation of the embedding function 'fun' is
                                needed? Depends on how sharp the function 'fun' is and the size of the matrix. For high
                                dimensional matrices "errors" in approximation can add up. So larger values of 'poly_order'
                                are recommended
        'boost' - How to approximate 'fun'? If nulls in the spectrum are important (which is typically the case when we
                                want to suppress noise), we recommend setting boost to 2 or 3 (as opposed to 1). In this
                                case, we obtain the polynomial approximation of fun(x) ^ (1/boost) and cascade the
                                 algorithm 'boost' number of times. See the paper in the README for details
        'sigma_max' - The spectral norm of the matrix 'mat' computed elsewhere. If this parameter is specified we do not
                                attempt to estimate the spectral norm of the matrix and hence the parameters
                                'spec_norm_est_dims', 'spec_norm_est_iter' and 'scale_up_spec_norm_est' are unused
        'spec_norm_est_dims' - Number of random starting vectors to use to estimate the spectral norm of the matrix
                                via power iteration
        'spec_norm_est_iter' - Number of iterates of power iteration for estimating the spectral norm (depends on how
                                clustered the singular values are around the spectral norm)
        'scale_up_spec_norm_est' - After estimating the spectral norm using power iteration (this is strictly a lower
                                bound on the spectral norm), we multiply it by a small number greater than 1 given by
                                scale_up_spec_norm_est (defaults to FastEmbed.SCALE_UP_SPEC_NORM_EST) to potentially
                                arrive at an upper bound. Since the algorithm assumes that we have an upper bound on the
                                spectral norm of the matrix, this parameter alongside 'spec_norm_est_dims',
                                'spec_norm_est_iter' may have to be tuned if the algorithm diverges.
                                Another option in those cases is to provide the spectral norm computed elsewhere
                                using the parameter sigma_max
        'n_jobs' - Number of cores to use in embedding computation (defaults to -1, which denotes all available cores)
        'normalize_fun' - if set to True (default), fun represents the scaling of the singular value relative to the
                                spectral norm (largest magnitude singular value) ie,
                                f(singularvector/spectralnorm) * singularvector; if set to False, we use fun as is on
                                the singularvalue to obtain its weight ie., f(singularvalue) * singularvector
         'verbose' - if set to True - prints details along the way; Defaults to False
    :param phi **** Optional ****
        how to weight polynomial approximation errors
        integrate(phi(x) * (fun(x) - poly_approx_of_fun(x))^2, -1, 1) (Defaults to phi(x) = 1.)
    :return: a tuple with embedding of (rows, cols) of size (#rows, embed_dims) and (#columns, embed_dims)
    """
    m = mat.shape[0]
    n = mat.shape[1]

    mat = sparse.csr_matrix(mat)

    # extra memory : 2*mem(mat)
    sym_mat = sparse.vstack([sparse.hstack([sparse.csr_matrix((n, n)), mat.transpose(copy=True)]),
                             sparse.hstack([mat, sparse.csr_matrix((m, m))])]).tocsr()

    def fun_prime(x):
        return fun(x) * (x >= 0) - fun(-x) * (x < 0)

    sym_embed = FastEmbed(sym_mat, fun_prime, embed_dims, config, phi)
    embed_rows = sym_embed.embed[range(n, m + n), :]
    embed_cols = sym_embed.embed[range(n), :]

    return embed_rows, embed_cols
