import tenseal as ts
import os

current_path = os.path.dirname(os.path.abspath(__file__))

def generate_and_save_keys():
    poly_modulus_degree = 8192 
    coeff_mod_bit_sizes = [60, 40, 40, 60]
    global_scale = 2**40

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes
    ) 
    context.generate_galois_keys()
    context.global_scale = global_scale

    with open(os.path.join(current_path, "ckks_full_context.bytes"), "wb") as f:
        if context.is_private():
            f.write(context.serialize(save_secret_key=True))

    public_context = context.copy()
    public_context.make_context_public()
    with open(os.path.join(current_path, "ckks_public.bytes"), "wb") as f:
        f.write(public_context.serialize())
    print("Keys have been saved to files: ckks_full_context.bytes and ckks_public.bytes.")

generate_and_save_keys()