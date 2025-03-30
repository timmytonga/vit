import torch
from projectors.utils import get_orthogonal_matrix


class OjaProjector:
    def __init__(self, rank, step_size=1e-5, do_qr=True, proj_type="oja"):
        self.rank = rank
        self.step_size = step_size  # step size eta_t for oja's algorithm
        self.ortho_matrix = None  # this is the projection matrix
        self.init_style = "rand" if proj_type == "ojar" else "svd"
        self.dtype = None  # this will be set by the grad
        self.do_qr = do_qr
        self.optim = None
        self.proj_type = proj_type

    def project(self, full_rank_grad: torch.Tensor, n_iter=None):
        # proj will be of shape n x k so final low_rank will be of shape m x k with rank k
        if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
            low_rank_grad = self.oja(full_rank_grad)
        else:  # n < m -- proj will be of shape m x k so final low_rank will be of shape k x n with rank k
            low_rank_grad = self.oja(full_rank_grad.t()).t()
        return low_rank_grad

    def project_back(self, low_rank_grad: torch.Tensor):
        if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix.t())
        else:
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        return full_rank_grad

    def oja(self, full_rank_grad: torch.Tensor):
        """
        Assume full_rank_grad is of shape m x n where m >= n. Should check before doing this.
        """
        assert full_rank_grad.shape[0] >= full_rank_grad.shape[1], "Must have dim0>=dim1 to use this fn"
        if self.ortho_matrix is None:  # initialization with svd
            self.ortho_matrix = self.get_init_mat(full_rank_grad, init_style=self.init_style)
            self.dtype = full_rank_grad.data.dtype  # set dtype
            self.get_optim()  # initialize optimizer
        else:
            self.ortho_matrix.grad = -full_rank_grad.t().matmul(full_rank_grad.matmul(self.ortho_matrix))
            self.optim.step()
            if self.do_qr:
                qr_data = self.ortho_matrix.data.float() if self.dtype != torch.float else self.ortho_matrix.data
                q, _ = torch.linalg.qr(qr_data, mode="reduced")
                self.ortho_matrix = q.to(self.dtype)
            self.ortho_matrix.grad = None
        low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix)
        return low_rank_grad

    def get_init_mat(self, full_rank_grad, init_style="svd"):
        """
        What to initialize the
        Args:
            full_rank_grad:
            init_style:

        Returns:

        """
        if init_style == "svd":
            ortho_mat = get_orthogonal_matrix(full_rank_grad, self.rank, type='right')  # k x n
            return ortho_mat.t()
        elif init_style == "rand":
            ortho_mat = torch.randn((full_rank_grad.shape[1], self.rank), dtype=full_rank_grad.dtype,
                                    device=full_rank_grad.device)
            return ortho_mat
        else:
            raise NotImplementedError

    def get_optim(self):
        if "adam" in self.proj_type:
            self.optim = torch.optim.AdamW([self.ortho_matrix], lr=self.step_size)
        else:
            self.optim = torch.optim.SGD([self.ortho_matrix], lr=self.step_size)
