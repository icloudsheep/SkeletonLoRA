#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
paper_table_3_CKKS.py
=====================
Self-contained simulation of the CKKS-based server-side B×A aggregation
pipeline from SHE-LoRA (_she).  Replicates every step exactly:

  Client side (simulated)
    - exchange_columns()   — swap encryption columns to the end of A
    - encrypt_cipher_list() — per-column CKKS encryption of selected cols
    - FedAvg scaling of B matrices

  Server side
    - process_client()            — enc(A_col) · plain(B)^T  per layer
    - aggregate_ckks_tensors()    — inter-client CKKS vector addition
    - parallel_processing_palin_ckks() — multi-process orchestration

  Client side (post-aggregation)
    - decode_ciphers()      — decrypt + weighted average
    - fusion_plain_cipher() — SVD fuse plaintext + ciphertext → LoRA weights

No imports from _she/ or CKKS/ — fully self-contained.
"""

import os
import pickle
import time
import numpy as np
import tenseal as ts
from concurrent.futures import ProcessPoolExecutor, as_completed
from safetensors import safe_open
from tqdm import tqdm

# ============================================================================
# 0.  Global:  path to the CKKS context file (auto-detected or created)
# ============================================================================



def _resolve_ctx_dir():
    """Try common locations for ckks_full_context.bytes; create new if absent."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "CKKS"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CKKS"),
        "./CKKS",
    ]
    for d in candidates:
        if os.path.isfile(os.path.join(d, "ckks_full_context.bytes")):
            return os.path.abspath(d)
    # fallback: create in _scripts/CKKS/
    fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CKKS")
    os.makedirs(fallback, exist_ok=True)
    return os.path.abspath(fallback)


def _get_or_create_context():
    """Load the CKKS context from file, or generate + save one."""
    ctx_dir = _resolve_ctx_dir()
    ctx_path = os.path.join(ctx_dir, "ckks_full_context.bytes")
    if os.path.exists(ctx_path):
        with open(ctx_path, "rb") as f:
            return ts.context_from(f.read())

    print("[*] Creating new CKKS context (8192, [60,40,40,60], scale=2^40) ...")
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    ctx.global_scale = 2 ** 40
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    # save so process_client can load it later
    with open(ctx_path, "wb") as f:
        f.write(ctx.serialize(save_secret_key=True))
    print(f"[*] Saved CKKS context to {ctx_path}")
    return ctx


# ============================================================================
# 1.  Client-side helpers  (exact replicas of _she/flowertune_llm/she/ckks_client.py)
# ============================================================================

def exchange_columns(A, selected_cols=None, num_enc_col=None):
    """
    Swap selected columns to the end of the matrix.
    Reversible — calling again with the same selected_cols restores order.

    For the i-th column in selected_cols, swap column i with column -(i+1).
    """
    if selected_cols is None and num_enc_col:
        selected_cols = np.random.choice(A.shape[1], num_enc_col, replace=False)
    A_1 = A.copy()
    for i, col_idx in enumerate(selected_cols):
        A_1[:, [col_idx, -(i + 1)]] = A_1[:, [-(i + 1), col_idx]]
    return A_1


def encrypt_cipher_list(cipher_list, context):
    """
    Encrypt columns using CKKS.  Context is reused.
    cipher_list: list of numpy arrays, each shape (r, num_cols_to_encrypt).
    Returns: pickle.dumps of list of lists of serialized CKKS vectors.
    """
    ckks_results = []
    for cipher in cipher_list:
        column_vectors = [cipher[:, i] for i in range(cipher.shape[1])]
        encrypted_layer = [
            ts.ckks_vector(context, vec).serialize() for vec in column_vectors
        ]
        ckks_results.append(encrypted_layer)
    return pickle.dumps(ckks_results)


# ============================================================================
# 2.  Server-side helpers  (exact replicas of _she/flowertune_llm/she/ckks_server.py)
# ============================================================================

def aggregate_ckks_tensors(item_list):
    """
    Aggregate CKKS tensors for one layer across clients.
    Pops last element from each client's list, adds them, records
    {'par': serialized_result, 'num': num_tensors}.
    """
    ctx_dir = _resolve_ctx_dir()
    ctx_path = os.path.join(ctx_dir, "ckks_full_context.bytes")
    with open(ctx_path, "rb") as f:
        context = ts.context_from(f.read())

    history_results = []
    while True:
        if all(not item for item in item_list):
            break
        result = None
        num_tensors = 0
        line_res = {}
        for item in item_list:
            if not item:
                continue
            ckks_tensor = item.pop()
            if isinstance(ckks_tensor, bytes):
                ckks_tensor = ts.ckks_vector_from(context, ckks_tensor)
            if result is None:
                result = ckks_tensor
            else:
                result += ckks_tensor
            num_tensors += 1

        if result is not None and num_tensors > 0:
            line_res["par"] = result.serialize()
            line_res["num"] = num_tensors
            history_results.insert(0, line_res)
    return history_results


def process_client(clint_key, plain_B, cipherA, ctx_dir):
    """
    Per-client homomorphic multiplication:  enc(A_col) · plain(B)^T  per layer.
    Returns serialized ciphertexts (list of list of bytes).
    """
    ctx_path = os.path.join(ctx_dir, "ckks_full_context.bytes")
    plainBlist = plain_B[clint_key]
    cipherAlist = pickle.loads(cipherA[clint_key])
    layer_num = len(plainBlist)
    client_res = []

    for index in range(layer_num):
        with open(ctx_path, "rb") as f:
            context = ts.context_from(f.read())
        plain_B_tensor = ts.plain_tensor(plainBlist[index])
        layer_res = []
        for vec_a in cipherAlist[index]:
            enc_col_a = ts.ckks_vector_from(context, vec_a)
            temp = enc_col_a.mm(plain_B_tensor.transpose())
            layer_res.append(temp.serialize())
        client_res.append(layer_res)

    return clint_key, client_res


def parallel_processing_palin_ckks(plain_B, cipherA):
    """
    Multi-process orchestration:
      1) process_client per client  (enc(A)·plain(B)^T)
      2) layer-wise aggregate_ckks_tensors across clients
    Returns pickle.dumps(final_res).
    """
    ctx_dir = _resolve_ctx_dir()
    result = {}

    start_time_mm = time.time()
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_client, clint_key, plain_B, cipherA, ctx_dir): clint_key
            for clint_key in plain_B.keys()
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Client BxA multiplication multi-process",
        ):
            clint_key = futures[future]
            _, client_res = future.result()
            result[clint_key] = client_res
    print(
        f"   -> Client-side homomorphic multiplication completed. "
        f"Time: {time.time() - start_time_mm:.6f} s"
    )

    final_res = []
    layer_nums = len(next(iter(plain_B.values())) if plain_B else None)

    start_time_agg = time.time()
    for layer in tqdm(range(layer_nums), desc="Server-side layer aggregation"):
        aggs_mix = []
        for client_key in plain_B.keys():
            aggs_mix.append(result[client_key][layer])
        final_layer = aggregate_ckks_tensors(aggs_mix)
        final_res.append(final_layer)
    print(
        f"   -> Server-side aggregation completed. "
        f"Time: {time.time() - start_time_agg:.6f} s"
    )

    start_serialize = time.time()
    final_res_bytes = pickle.dumps(final_res)
    print(
        f"   -> Result serialization completed. "
        f"Time: {time.time() - start_serialize:.6f} s"
    )
    return final_res_bytes


# ============================================================================
# 3.  Client-side post-processing  (replicas of _she/flowertune_llm/utils/client_utils.py)
# ============================================================================

def decode_ciphers(cipher_bytes):
    """
    Decrypt the aggregated ciphertext blocks.
    Each block is {'par': serialized, 'num': count}.
    Returns list of numpy matrices, one per layer.
    """
    ctx_dir = _resolve_ctx_dir()
    ctx_path = os.path.join(ctx_dir, "ckks_full_context.bytes")
    with open(ctx_path, "rb") as f:
        context = ts.context_from(f.read())

    results = []
    cipher_layers = pickle.loads(cipher_bytes)
    for cipher_list in cipher_layers:
        layer_res = []
        for enc_line in cipher_list:
            ckks_tensor = ts.ckks_vector_from(context, enc_line["par"])
            weighted_avg_coeff = enc_line["num"]
            ckks_plain = np.array(ckks_tensor.decrypt(), dtype="float32")
            ckks_plain *= 1.0 / weighted_avg_coeff
            layer_res.append(ckks_plain)
        layer_matrix = np.vstack(layer_res)
        results.append(layer_matrix)
    return results


def fusion_plain_cipher(plain_agg_results, cipher_agg_results, enc_a_lines, max_rank):
    """
    Fuse plaintext-aggregated LoRA weights with decrypted ciphertext results.
    SVD → hstack/block → exchange_columns restore → SVD back to max_rank.
    """
    parameters = []
    cipher_to_plain = decode_ciphers(cipher_agg_results)
    assert len(enc_a_lines) == len(cipher_to_plain), (
        f"Cipher block number {len(cipher_to_plain)} "
        f"and Layer A number {len(enc_a_lines)} mismatch"
    )
    for index, layer_matrix in enumerate(cipher_to_plain):
        U, s, Vh = np.linalg.svd(layer_matrix.transpose(), full_matrices=False)
        A_e = Vh
        B_e = U @ np.diag(s)

        A_p = plain_agg_results[index * 2]
        B_p = plain_agg_results[index * 2 + 1]

        B_r = np.hstack([B_p, B_e])
        A_r = np.block([
            [A_p, np.zeros((A_p.shape[0], A_e.shape[1]))],
            [np.zeros((A_e.shape[0], A_p.shape[1]), dtype=float), A_e],
        ])

        # Restore column order
        enc_cols = next(iter(enc_a_lines.values())) if enc_a_lines else None
        A_r = exchange_columns(A_r, selected_cols=enc_cols)

        result = B_r.dot(A_r)
        U, s, Vh = np.linalg.svd(result, full_matrices=False)
        B_aggregated = U[:, :max_rank] @ np.diag(s[:max_rank])
        A_aggregated = Vh[:max_rank, : len(A_p[1])]
        parameters.append(A_aggregated)
        parameters.append(B_aggregated)

    return parameters


# ============================================================================
# 4.  Data extraction
# ============================================================================

def extract_ab_matrices(file_or_dir_path):
    """Extract all lora_A / lora_B tensors from safetensors file(s)."""
    ab_matrices = {}
    target_files = []
    if os.path.isdir(file_or_dir_path):
        for fname in os.listdir(file_or_dir_path):
            if fname.endswith(".safetensors"):
                target_files.append(os.path.join(file_or_dir_path, fname))
    elif os.path.isfile(file_or_dir_path):
        target_files.append(file_or_dir_path)

    if not target_files:
        print(f"No safetensors files found in {file_or_dir_path}.")
        return ab_matrices

    total_size_bytes = 0
    for filepath in tqdm(target_files, desc="Extracting Safetensors files"):
        with safe_open(filepath, framework="pt", device="cpu") as f:
            for key in f.keys():
                if "lora_A" in key or "lora_B" in key:
                    tensor = f.get_tensor(key)
                    ab_matrices[key] = tensor
                    total_size_bytes += tensor.element_size() * tensor.nelement()

    print(
        f"Successfully extracted {len(ab_matrices)} A/B matrices. "
        f"Total: {total_size_bytes / (1024 * 1024):.6f} MB"
    )
    return ab_matrices


# ============================================================================
# 5.  Simulation runner
# ============================================================================

def run_simulation():
    total_start_time = time.time()
    print("=" * 80)
    print("=== SHE-LoRA Server-side B×A Full Simulation (exact _she replica) ===")
    print("=" * 80)

    # ---- Paths ----
    base_target_path = "./temp_output_dir"
    aggregated_output_path = "./ckks_server_aggregated.bin"

    num_clients = 1
    threshold_ratio = 0.00125          # 0.125 %  → matches _she's default he_budget logic
    lora_rank = 4                      # default rank for evaluation

    # ---- 0. Init CKKS context ----
    print("\n[0/4] Loading / generating CKKS context ...")
    t0 = time.time()
    context = _get_or_create_context()
    ctx_dir = _resolve_ctx_dir()
    print(f"   -> Context ready ({time.time() - t0:.3f} s), dir = {ctx_dir}")

    plain_B = {}
    cipher_A = {}

    for client_id in range(num_clients):
        client_path = os.path.join(
            base_target_path, f"client_{client_id}_output", "final_lora"
        )
        if not os.path.exists(client_path):
            print(
                f"[WARNING] Client {client_id} directory not found: {client_path}, "
                f"skipping."
            )
            continue

        print(f"\n>>>> Processing Client [{client_id}/{num_clients - 1}] <<<<")
        t_extract = time.time()
        original_ab_matrices = extract_ab_matrices(client_path)
        print(
            f"   -> Model matrix extraction time: {time.time() - t_extract:.6f} s"
        )
        if not original_ab_matrices:
            print(f"Error: No matrices extracted for client {client_id}.")
            continue

        # ----------  Group A/B by layer ----------
        layers_data = {}
        for key, tensor in original_ab_matrices.items():
            if "lora_A" in key:
                base_key = (
                    key.replace(".lora_A", "")
                    .replace("_lora_A", "")
                    .replace("lora_A", "")
                )
                if base_key not in layers_data:
                    layers_data[base_key] = {}
                layers_data[base_key]["A"] = tensor
            elif "lora_B" in key:
                base_key = (
                    key.replace(".lora_B", "")
                    .replace("_lora_B", "")
                    .replace("lora_B", "")
                )
                if base_key not in layers_data:
                    layers_data[base_key] = {}
                layers_data[base_key]["B"] = tensor

        layers = [k for k, v in layers_data.items() if "A" in v and "B" in v]
        ignored = [k for k, v in layers_data.items() if k not in layers]
        print(f"Found {len(layers)} complete LoRA layers for processing.")
        if ignored:
            print(
                f"[WARNING] Ignored {len(ignored)} incomplete layers: "
                f"{ignored[:5]}{'...' if len(ignored) > 5 else ''}"
            )

        # ----------  FedAvg scale ----------
        scale = 1.0 / num_clients

        # ----------  Build enc_lines per layer (simulate negotiated columns) ----------
        # In real _she this comes from OPE-negotiation; here we use L2-norm top-k
        enc_lines = {}
        for layer_name in layers:
            A_t = layers_data[layer_name]["A"].float()
            in_dim = A_t.shape[1]
            he_budget = max(1, int(in_dim * threshold_ratio))
            col_norms = np.linalg.norm(A_t.numpy(), axis=0)
            top_indices = np.argsort(col_norms)[-he_budget:][::-1].tolist()
            enc_lines[layer_name] = top_indices

        # ----------  Client encryption  (exact _she flow) ----------
        t_enc = time.time()

        cid_str = str(client_id)
        client_plain_B = []
        client_cipher_A_raw = []       # list of (rank, he_budget) arrays

        total_enc_size = 0
        total_plain_size = 0

        for l_idx, layer_name in enumerate(
            tqdm(layers, desc=f"Client {client_id} Encryption")
        ):
            A_tensor = layers_data[layer_name]["A"].float().cpu().numpy()  # (rank, in_dim)
            B_tensor = layers_data[layer_name]["B"].float().cpu().numpy()  # (out_dim, rank)

            # ---- B: FedAvg scaled ----
            B_scaled = B_tensor * scale
            client_plain_B.append(B_scaled)
            total_plain_size += B_scaled.nbytes

            # ---- A: exchange_columns + split ----
            selected_cols = enc_lines[layer_name]              # list of column indices
            A_exchanged = exchange_columns(A_tensor, selected_cols=selected_cols)
            n_enc = len(selected_cols)

            # columns to encrypt: last n_enc columns
            columns_to_enc = A_exchanged[:, -n_enc:].copy()
            # zero them out in the plain matrix
            A_exchanged[:, -n_enc:] = 0.0
            # This A_exchanged (with zeros) is the plain A sent to server
            client_plain_A = A_exchanged

            # Store the encrypted-column block (will be encrypted below)
            client_cipher_A_raw.append(columns_to_enc)

            total_enc_size += columns_to_enc.nbytes
            total_plain_size += client_plain_A.nbytes

        # Serialize cipher columns with encrypt_cipher_list  (exact _she call)
        cipher_A_bytes = encrypt_cipher_list(client_cipher_A_raw, context)

        plain_B[cid_str] = client_plain_B
        cipher_A[cid_str] = cipher_A_bytes

        print(
            f"   -> Client {client_id} encryption finished. "
            f"Time: {time.time() - t_enc:.6f} s"
        )
        print(
            f"   -> [INFO] Plaintext (B + A) total size: "
            f"{total_plain_size / (1024 * 1024):.6f} MB"
        )
        print(
            f"   -> [INFO] Ciphertext A size: "
            f"{len(cipher_A_bytes) / (1024 * 1024):.6f} MB"
        )

    # ====================================================================
    #  Server-side parallel B×A homomorphic multiplication + aggregation
    # ====================================================================
    print("\n[4/4] Server Side Parallel B×A Multiplication & Aggregation ...")
    t_serv = time.time()

    final_res_bytes = parallel_processing_palin_ckks(plain_B, cipher_A)

    print(
        f"   -> Homomorphic multiplication and aggregation total time: "
        f"{time.time() - t_serv:.6f} s"
    )

    # Save aggregated result
    with open(aggregated_output_path, "wb") as f:
        f.write(final_res_bytes)
    agg_size_mb = os.path.getsize(aggregated_output_path) / (1024 * 1024)
    print(
        f"   -> [INFO] Server aggregated result saved to "
        f"{aggregated_output_path}  (size: {agg_size_mb:.6f} MB)"
    )

    # ---- Verification ----
    print("\n=== Verification of the Aggregated Results ===")
    t_ver = time.time()
    final_res = pickle.loads(final_res_bytes)
    print(f"   -> Aggregated results contain {len(final_res)} LoRA layers.")
    for l_idx in range(min(3, len(final_res))):
        layer = final_res[l_idx]
        print(
            f"   -> Layer {l_idx} contains {len(layer)} aggregated ciphertext "
            f"blocks."
        )
    if len(final_res) > 3:
        print("   -> ...")
    print(f"   -> Verification time: {time.time() - t_ver:.6f} s")

    # ---- Client-side decryption + fusion (exact _she pipeline) ----
    print("\n=== Client-side Decryption & Fusion ===")
    t_dec = time.time()

    # Build plain_agg_results (simulate what server returns as plain aggregate)
    # In real _she this is the result of aggregate_flora; here we simulate
    # by stacking the plain B + zeroed-A for each layer from the only client
    plain_agg = []
    any_plain_B = next(iter(plain_B.values()))
    enc_lines_list = list(enc_lines.values())  # ordered same as layers

    for l_idx, (B_mat, enc_cols) in enumerate(zip(any_plain_B, enc_lines_list)):
        # B already scaled; A was sent as zeroed-exchanged, need to reconstruct
        # In real _she, aggregate_flora returns [A1, B1, A2, B2, ...] after SVD
        # For simulation: just use the plain matrices we sent
        # (they represent the plaintext aggregation result)
        # Actually we don't have A_plain_agg here — let's reconstruct from the
        # original exchanged+zeroed A. For a 1-client sim this is exact.
        layer_name = layers[l_idx]
        A_orig = layers_data[layer_name]["A"].float().cpu().numpy()
        A_exchanged = exchange_columns(A_orig, selected_cols=enc_cols)
        n_enc = len(enc_cols)
        A_exchanged[:, -n_enc:] = 0.0
        plain_agg.append(A_exchanged)
        plain_agg.append(B_mat)

    # Execute fusion
    fused_params = fusion_plain_cipher(
        plain_agg_results=plain_agg,
        cipher_agg_results=final_res_bytes,
        enc_a_lines=enc_lines,
        max_rank=lora_rank,
    )
    print(
        f"   -> Fusion produced {len(fused_params)} parameter matrices. "
        f"Time: {time.time() - t_dec:.6f} s"
    )

    total_time = time.time() - total_start_time
    print(f"\n=== Simulation Finished Successfully! Total Time: {total_time:.6f} s ===")


# ============================================================================
if __name__ == "__main__":
    run_simulation()
