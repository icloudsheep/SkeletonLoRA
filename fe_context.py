"""CKKS 密钥上下文工具。

提供两类 context：
  - 私钥 context（含 secret key）：唯一能解密，交给持私钥的客户端解密方。
  - 公开 context（去 secret key）：能加密、能同态加乘，无法解密，交给服务端。

密钥边界是 HE 隐私保证的基础：服务端只拿公开 context，全程无法看到明文。
main 负责建立并按角色分发这两个 context，本模块只提供构造/派生能力。
"""

import tenseal as ts


def create_secret_context(poly_modulus_degree, coeff_mod_bit_sizes, global_scale,
                          galois=False):
    """创建含私钥的 CKKS context。

    galois 旋转密钥按需生成：外积法全程逐元素 ct×ct 与 ct×plain，不涉及槽位旋转，
    传 False 省去 galois key 的生成开销与体积；内积法要靠 matmul/dot 做跨槽位求和，
    必须传 True 生成 galois key，否则 matmul 会报「doesn't hold a Galois keys」。

    :param poly_modulus_degree: 多项式模数次数，决定可打包槽位数（= 次数/2）
    :param coeff_mod_bit_sizes: 系数模数链各素数比特宽度，决定可用乘法深度
    :param global_scale: CKKS 全局缩放因子，影响定点编码精度
    :param galois: 是否生成 galois 旋转密钥（内积法跨槽位求和所需）
    :return: 含 secret/public/relin(/galois) key 的私钥 context
    """
    ctx = ts.context(ts.SCHEME_TYPE.CKKS,
                     poly_modulus_degree=poly_modulus_degree,
                     coeff_mod_bit_sizes=coeff_mod_bit_sizes)
    ctx.global_scale = global_scale
    if galois:
        ctx.generate_galois_keys()
    return ctx


def derive_public_context(secret_ctx):
    """从私钥 context 派生去掉私钥的公开 context。

    公开 context 保留 public/relin 等 evaluation key，可加密、可同态加乘，
    但 is_private() 为 False、无法解密。用于分发给服务端。

    :param secret_ctx: 含私钥的 context
    :return: 去私钥的公开 context
    """
    pub = secret_ctx.copy()
    pub.make_context_public()
    return pub
