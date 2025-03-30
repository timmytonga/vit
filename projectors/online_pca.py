import torch
import transformers

from projectors.utils import get_orthogonal_matrix


def online_pca(self, full_rank_grad, update_proj_stepsize_ratio):
    if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
        if self.ortho_matrix is None:
            self.ortho_matrix = get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
            get_optim(self)
        else:
            with torch.enable_grad():
                self.ortho_matrix.requires_grad = True
                self.ortho_matrix.grad = None
                projection = self.ortho_matrix.t() @ self.ortho_matrix
                normalized_full_rank_grad = full_rank_grad / torch.norm(full_rank_grad)
                loss = torch.norm(normalized_full_rank_grad @ projection - normalized_full_rank_grad) ** 2
                loss.backward()
                update_proj_stepsize = 1 / self.update_proj_gap * update_proj_stepsize_ratio
                for group in self.ortho_matrix_optim.param_groups:
                    group["lr"] = update_proj_stepsize
                self.ortho_matrix.requires_grad = False
        low_rank_grad = torch.matmul(full_rank_grad, self.ortho_matrix.t())
    else:
        if self.ortho_matrix is None:
            self.ortho_matrix = get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
            get_optim(self)

        else:
            with torch.enable_grad():
                self.ortho_matrix.requires_grad = True
                self.ortho_matrix.grad = None
                projection = self.ortho_matrix @ self.ortho_matrix.t()
                normalized_full_rank_grad = full_rank_grad / torch.norm(full_rank_grad)
                loss = torch.norm(projection @ normalized_full_rank_grad - normalized_full_rank_grad) ** 2
                loss.backward()
                update_proj_stepsize = 1 / self.update_proj_gap * update_proj_stepsize_ratio
                for group in self.ortho_matrix_optim.param_groups:
                    group["lr"] = update_proj_stepsize
                self.ortho_matrix.requires_grad = False
        low_rank_grad = torch.matmul(self.ortho_matrix.t(), full_rank_grad)
    return low_rank_grad


def get_optim(self):
    if "adafactor" in self.proj_type:
        self.ortho_matrix_optim = transformers.optimization.Adafactor(
            [self.ortho_matrix],
            lr=1 / self.update_proj_gap,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    elif "sgd" in self.proj_type:
        self.ortho_matrix_optim = torch.optim.SGD([self.ortho_matrix], lr=1 / self.update_proj_gap)
    else:
        self.ortho_matrix_optim = torch.optim.AdamW([self.ortho_matrix], lr=1 / self.update_proj_gap)
