import torch


def get_orthogonal_matrix(weights, rank, type):
    module_params = weights

    if module_params.data.dtype != torch.float:
        float_data = False
        original_type = module_params.data.dtype
        original_device = module_params.data.device
        matrix = module_params.data.float()
    else:
        float_data = True
        matrix = module_params.data

    U, s, Vh = torch.linalg.svd(matrix, full_matrices=False)

    # make the smaller matrix always to be orthogonal matrix
    if type == 'right':
        B = Vh[:rank, :]

        if not float_data:
            B = B.to(original_device).type(original_type)
        return B
    elif type == 'left':
        A = U[:, :rank]
        if not float_data:
            A = A.to(original_device).type(original_type)
        return A
    elif type == 'full':
        A = U[:, :rank]
        B = Vh[:rank, :]
        if not float_data:
            A = A.to(original_device).type(original_type)
            B = B.to(original_device).type(original_type)
        return [A, B]
    else:
        raise ValueError('type should be left, right or full')
