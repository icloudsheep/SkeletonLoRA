
import tenseal as ts
import numpy as np
import os
import torch

_CACHED_CONTEXT = None
_CACHED_CONTEXT_PATH = None


def _get_context(context_path):
    global _CACHED_CONTEXT, _CACHED_CONTEXT_PATH

    if _CACHED_CONTEXT is not None and _CACHED_CONTEXT_PATH == context_path:
        return _CACHED_CONTEXT

    if not os.path.exists(context_path):
        raise FileNotFoundError(f"TenSEAL context file not found at: {context_path}")

    with open(context_path, "rb") as f:
        context = ts.context_from(f.read())
        _CACHED_CONTEXT = context
        _CACHED_CONTEXT_PATH = context_path
        print(f"[CKKS] Loaded context from {context_path}")
        return context

def get_context(context_path="./tools/ckks_full_context.bytes"):
    return _get_context(context_path)


def encrypt_matrix_blocks(matrix, rank=4, max_slots=4096, context_path="./CKKS/ckks_full_context.bytes"):
    context = _get_context(context_path)

    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy()

    if len(matrix.shape) == 1:
        matrix = matrix.reshape(-1, rank)
    
    N, current_rank = matrix.shape
    assert current_rank == rank, f"Matrix column size {current_rank} does not match expected rank {rank}."

    chunk_size = max_slots // rank  
    cipher_list = []

    for i in range(0, N, chunk_size):
        block = matrix[i:i + chunk_size, :]
        
        flat_block = block.flatten().tolist()
        
        if len(flat_block) < max_slots:
            flat_block.extend([0.0] * (max_slots - len(flat_block)))

        enc_vec = ts.ckks_vector(context, flat_block)
        cipher_list.append(enc_vec.serialize())

    return cipher_list


def decrypt_matrix_blocks(cipher_list, original_shape, rank=4, max_slots=4096, context_path="./CKKS/ckks_full_context.bytes"):
    if not cipher_list:
        return np.array([])

    context = _get_context(context_path)
    N, _ = original_shape
    chunk_size = max_slots // rank

    decrypted_data = []

    for cipher_bytes in cipher_list:
        enc_vec = ts.ckks_vector_from(context, cipher_bytes)
        dec_vec = enc_vec.decrypt()
        decrypted_data.extend(dec_vec)

    total_elements = N * rank
    matrix = np.array(decrypted_data[:total_elements]).reshape(original_shape)
    
    return matrix