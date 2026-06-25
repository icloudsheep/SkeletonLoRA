# Low-Rank Skeleton Decryption Correctness Test

当前测试对象为 4 个客户端。每个客户端持有一组 LoRA 因子：

$$
B_i\in \mathbb R^{3200\times 4},\qquad A_i\in \mathbb R^{4\times 3200}.
$$

每个客户端的本地更新为：

$$
\Delta W_i=B_iA_i\in \mathbb R^{3200\times 3200}.
$$

聚合后的全局更新为：

$$
\Delta W=\sum_{i=1}^{4}B_iA_i.
$$

由于每个 (B_iA_i) 的秩最多为 4，因此：

$$
\operatorname{rank}(\Delta W)
\le
\sum_{i=1}^{4}\operatorname{rank}(B_iA_i)
\le
4\times 4=
16.
$$

因此，在该实验设置下，聚合矩阵 $\Delta W$ 的秩最多为 16。我们不固定使用 rank 16，而是将 skeleton rank 设置为变量 (r)，从 (r=4) 开始逐渐增加，测试在多大的 r 下可以正确恢复完整聚合矩阵。

---

## Test Goal

本实验主要包含两个目标：

1. **Correctness Verification**：验证只解密 r 行和 r 列是否可以恢复完整的聚合矩阵。
2. **Time Analysis**：记录不同 r下的解密时间。

正确性验证可以借助明文完成。具体来说，我们先在明文端直接计算：

$$
\Delta W_{\mathsf{plain}}=\sum_{i=1}^{4}B_iA_i,
$$

然后将 skeleton decryption 恢复得到的矩阵：

$$
\Delta W_{\mathsf{rec}}
$$

与明文结果进行比较。如果二者一致，或者相对误差低于预设阈值，则说明当前 (r) 下的 skeleton reconstruction 是正确的。

---

## Variable-Rank Skeleton Decryption

令 skeleton rank 为变量：

$$
r\in{4,5,6,\ldots,16}.
$$

对于每一个候选 r，执行如下流程。

### Step 1: Select Row and Column Indices

选择 r 个 row index：

$$
I_r\subseteq[3200],\qquad |I_r|=r,
$$

以及 r 个 column index：

$$
J_r\subseteq[3200],\qquad |J_r|=r.
$$

### Step 2: Decrypt the Intersection Block

先只解密交叉块：

$$
M_r=\Delta W[I_r,J_r]\in \mathbb R^{r\times r}.
$$

然后检查：

$$
\operatorname{rank}(M_r)=r.
$$

如果 (M_r) 不满秩，则说明当前选择的行列无法张成目标子空间。此时可以重新选择 (I_r,J_r)，或者继续增大 (r)。

### Step 3: Decrypt the Selected Columns and Rows

如果 (M_r) 满秩，则继续解密对应的 (r) 列：

$$
C_r=\Delta W[:,J_r]\in \mathbb R^{3200\times r},
$$

以及对应的 (r) 行：

$$
R_r=\Delta W[I_r,:]\in \mathbb R^{r\times 3200}.
$$

### Step 4: Reconstruct the Full Aggregated Matrix

使用 skeleton reconstruction 恢复完整矩阵：

$$
\Delta W_{\mathsf{rec}}=C_rM_r^{-1}R_r.
$$

也可以写成低秩分解形式：

$$
U_r=C_r,\qquad V_r=M_r^{-1}R_r,
$$

因此：

$$
\Delta W_{\mathsf{rec}}=U_rV_r.
$$

其中：

$$
U_r\in\mathbb R^{3200\times r},\qquad V_r\in\mathbb R^{r\times 3200}.
$$

### Step 5: Verify Correctness Using Plaintext Aggregation

在明文端直接计算 baseline：

$$
\Delta W_{\mathsf{plain}}=\sum_{i=1}^{4}B_iA_i.
$$

然后比较：

$$
\Delta W_{\mathsf{rec}}\stackrel{?}{=}\Delta W_{\mathsf{plain}}.
$$

实际实现中可以使用相对 Frobenius 范数误差：

$$
\epsilon_r=
\frac{|\Delta W_{\mathsf{rec}}-\Delta W_{\mathsf{plain}}|*F}
{|\Delta W*{\mathsf{plain}}|_F}.
$$

如果：

$$
\epsilon_r\le \tau,
$$

例如 (\tau=10^{-5}) 或 (10^{-6})，则认为当前 (r) 下恢复正确。

---

## Test Loop

```text
for r = 4, 5, 6, ..., 16:

    1. Select r rows I_r and r columns J_r.

    2. Decrypt the intersection block:
           M_r = DeltaW[I_r, J_r].

    3. Check rank(M_r).

       If rank(M_r) < r:
           retry with another I_r, J_r,
           or continue to the next r.

    4. Decrypt selected columns:
           C_r = DeltaW[:, J_r].

    5. Decrypt selected rows:
           R_r = DeltaW[I_r, :].

    6. Reconstruct:
           DeltaW_rec = C_r * inverse(M_r) * R_r.

    7. Compute plaintext baseline:
           DeltaW_plain = sum_i B_i * A_i.

    8. Compute relative error:
           error = ||DeltaW_rec - DeltaW_plain||_F / ||DeltaW_plain||_F.

    9. Record:
           r,
           rank(M_r),
           error,
           correctness result,
           decryption time.
```

---

