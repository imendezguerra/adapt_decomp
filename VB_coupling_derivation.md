# V→B Coupling Correction: Full Derivation

This document derives the first-order correction that must be applied to the
separation matrix **B** after each whitening update to **V**, so that the source
estimates remain frame-consistent between the two updates.

---

## 1. Notation

| Symbol | Shape | Description |
|--------|-------|-------------|
| $N$ | scalar | Number of samples in the current FIFO window |
| $D$ | scalar | Extended channel dimension ($\text{channels} \times \text{ext\_fact}$) |
| $M$ | scalar | Number of motor units |
| $\mathbf{X}$ | $N \times D$ | Zero-mean extended EMG batch |
| $\mathbf{V}$ | $D \times D$ | Whitening matrix (current, before update) |
| $\mathbf{V}'$ | $D \times D$ | Updated whitening matrix |
| $\mathbf{B}$ | $M \times D$ | Separation matrix (current) |
| $\mathbf{B}'$ | $M \times D$ | Corrected separation matrix (to be found) |
| $\mathbf{Z}$ | $N \times D$ | Whitened batch under $\mathbf{V}$ |
| $\mathbf{Z}'$ | $N \times D$ | Whitened batch under $\mathbf{V}'$ |
| $\mathbf{Y}$ | $N \times M$ | Source estimates under $(\mathbf{V}, \mathbf{B})$ |
| $\mathbf{Y}'$ | $N \times M$ | Source estimates under $(\mathbf{V}', \mathbf{B}')$ |
| $\mathbf{C}_X$ | $D \times D$ | Sample covariance of $\mathbf{X}$: $\mathbf{C}_X = \mathbf{X}^\top\mathbf{X}/N$ |
| $\mathbf{R}_Z$ | $D \times D$ | Sample covariance of $\mathbf{Z}$: $\mathbf{R}_Z = \mathbf{Z}^\top\mathbf{Z}/N = \mathbf{V}\mathbf{C}_X\mathbf{V}^\top$ |
| $\mathbf{I}$ | $D \times D$ | Identity matrix |
| $\mathbf{S}$ | $D \times D$ | Gradient direction matrix (symmetric); defined in §4 |
| $\Delta\mathbf{V}_\text{raw}$ | $D \times D$ | Natural-gradient step before trust-region clip |
| $\Delta\mathbf{V}$ | $D \times D$ | Clipped whitening step actually applied |
| $\Delta\mathbf{B}_\text{coupling}$ | $M \times D$ | Coupling correction to $\mathbf{B}$ |
| $\mathbf{A}$ | $D \times D$ | Relative change of $\mathbf{V}$: $\mathbf{A} = \Delta\mathbf{V}\,\mathbf{V}^{-1}$ |
| $\eta_v$ | scalar | Whitening learning rate |
| $e_v$ | scalar | Whitening error (scalar KL deviation) |
| $c$ | scalar | Trust-region clip scale factor ($0 < c \leq 1$) |
| $\mathbf{\Gamma}$ | $D \times D$ | Coupling matrix: $\mathbf{\Gamma} = c\,\eta_v\,e_v\,\mathbf{S}$ |

---

## 2. The Forward Model

The source estimation pipeline applies two linear transformations in sequence.

**Whitening** maps the raw extended EMG to a decorrelated representation:

$$\mathbf{Z} = \mathbf{X}\mathbf{V}^\top \tag{1}$$

**Source estimation** projects the whitened signal onto $M$ separation vectors:

$$\mathbf{Y} = \mathbf{Z}\mathbf{B}^\top \tag{2}$$

Substituting (1) into (2), the combined mapping from raw EMG to sources is:

$$\mathbf{Y} = \mathbf{X}\mathbf{V}^\top\mathbf{B}^\top \tag{3}$$

---

## 3. Effect of the V Update on the Whitened Signal

The whitening matrix is updated by an additive step:

$$\mathbf{V}' = \mathbf{V} + \Delta\mathbf{V} \tag{4}$$

The new whitened batch is:

$$\mathbf{Z}' = \mathbf{X}(\mathbf{V}')^\top = \mathbf{X}(\mathbf{V} + \Delta\mathbf{V})^\top = \mathbf{X}\mathbf{V}^\top + \mathbf{X}\Delta\mathbf{V}^\top \tag{5}$$

Using (1) to identify the first term:

$$\mathbf{Z}' = \mathbf{Z} + \Delta\mathbf{Z}, \qquad \Delta\mathbf{Z} = \mathbf{X}\Delta\mathbf{V}^\top \tag{6}$$

To rewrite $\Delta\mathbf{Z}$ in terms of $\mathbf{Z}$, invert (1) to express $\mathbf{X}$:

$$\mathbf{X} = \mathbf{Z}(\mathbf{V}^\top)^{-1} = \mathbf{Z}\mathbf{V}^{-\top} \tag{7}$$

Substituting (7) into $\Delta\mathbf{Z} = \mathbf{X}\Delta\mathbf{V}^\top$:

$$\Delta\mathbf{Z} = \mathbf{Z}\mathbf{V}^{-\top}\Delta\mathbf{V}^\top = \mathbf{Z}\left(\Delta\mathbf{V}\mathbf{V}^{-1}\right)^\top \tag{8}$$

Define the **relative change matrix**:

$$\mathbf{A} \;=\; \Delta\mathbf{V}\,\mathbf{V}^{-1} \tag{9}$$

so that (8) and (5) become:

$$\Delta\mathbf{Z} = \mathbf{Z}\mathbf{A}^\top \tag{10}$$

$$\mathbf{Z}' = \mathbf{Z}\left(\mathbf{I} + \mathbf{A}^\top\right) \tag{11}$$

Equation (11) shows that the V update acts on the whitened signal as a right-multiplication by $(\mathbf{I} + \mathbf{A}^\top)$, which is a first-order frame change.

---

## 4. Consistency Constraint

We require that the source estimates do not change between the pre-update and post-update parameterisations:

$$\mathbf{Y}' = \mathbf{Z}'\left(\mathbf{B}'\right)^\top \overset{!}{=} \mathbf{Y} = \mathbf{Z}\mathbf{B}^\top \tag{12}$$

Substituting (11) into the left side of (12):

$$\mathbf{Z}\left(\mathbf{I} + \mathbf{A}^\top\right)\left(\mathbf{B}'\right)^\top = \mathbf{Z}\mathbf{B}^\top \tag{13}$$

Rearranging:

$$\mathbf{Z}\left[\left(\mathbf{I} + \mathbf{A}^\top\right)\left(\mathbf{B}'\right)^\top - \mathbf{B}^\top\right] = \mathbf{0} \tag{14}$$

**Full-rank argument.** The FIFO is designed with $N_\text{fifo} \geq D$ samples, and Tikhonov shrinkage with parameter $s > 0$ ensures $\mathbf{R}_Z$ is positive definite, i.e. $\mathbf{Z}$ has full column rank $D$. Therefore the null space of $\mathbf{Z}$ (acting as a left-multiplier on $D \times M$ matrices) is $\{\mathbf{0}\}$, and (14) implies:

$$\left(\mathbf{I} + \mathbf{A}^\top\right)\left(\mathbf{B}'\right)^\top = \mathbf{B}^\top \tag{15}$$

Solving for $\mathbf{B}'$:

$$\left(\mathbf{B}'\right)^\top = \left(\mathbf{I} + \mathbf{A}^\top\right)^{-1}\mathbf{B}^\top \tag{16}$$

$$\mathbf{B}' = \mathbf{B}\left(\mathbf{I} + \mathbf{A}^\top\right)^{-\top} = \mathbf{B}\left(\mathbf{I} + \mathbf{A}\right)^{-1} \tag{17}$$

---

## 5. First-Order Approximation

The trust-region clip (§7) ensures $\|\Delta\mathbf{V}\|_F \leq r\|\mathbf{V}\|_F$ for a small constant $r$ (`max_rel_delta_v`). Consequently $\|\mathbf{A}\|$ is small and we can expand (17) to first order using:

$$(\mathbf{I} + \mathbf{A})^{-1} \approx \mathbf{I} - \mathbf{A} + O(\|\mathbf{A}\|^2) \tag{18}$$

The relative error of this approximation is $O(\|\mathbf{A}\|^2)$, controlled by `max_rel_delta_v`$^2$. Substituting (18) into (17):

$$\mathbf{B}' \approx \mathbf{B}(\mathbf{I} - \mathbf{A}) = \mathbf{B} - \mathbf{B}\mathbf{A} \tag{19}$$

The coupling correction is therefore:

$$\Delta\mathbf{B}_\text{coupling} = \mathbf{B}' - \mathbf{B} \approx -\mathbf{B}\mathbf{A} \tag{20}$$

Substituting (9) for $\mathbf{A}$:

$$\Delta\mathbf{B}_\text{coupling} \approx -\mathbf{B}\,\Delta\mathbf{V}\,\mathbf{V}^{-1} \tag{21}$$

Expression (21) still contains $\mathbf{V}^{-1}$, which is expensive ($O(D^3)$) to compute online. The next section shows how the natural gradient structure eliminates it.

---

## 6. Elimination of $\mathbf{V}^{-1}$ via the Natural Gradient Structure

The whitening update is a **natural-gradient** step. In both modes, the unclipped step takes the form:

$$\Delta\mathbf{V}_\text{raw} = -\eta_v\,e_v\,\mathbf{S}\,\mathbf{V} \tag{22}$$

where $\mathbf{S}$ is the mode-specific direction matrix:

$$\mathbf{S} = \begin{cases} \mathbf{R}_Z - \mathbf{I} & \text{(kl\_to\_identity)} \\ \mathbf{R}_{Z,\text{cal}}^{-1}\mathbf{R}_Z - \mathbf{I} & \text{(kl\_to\_cal)} \end{cases} \tag{23}$$

Both forms of $\mathbf{S}$ are **symmetric** because $\mathbf{R}_Z$ is symmetrised by construction ($\mathbf{R}_Z \leftarrow \tfrac{1}{2}(\mathbf{R}_Z + \mathbf{R}_Z^\top)$) at every FIFO update, and $\mathbf{R}_{Z,\text{cal}}^{-1}\mathbf{R}_Z$ is the product of two symmetric matrices sharing the same eigenbasis when $\mathbf{R}_{Z,\text{cal}}$ is positive definite (Cholesky factorisation argument):

$$\mathbf{S}^\top = \mathbf{S} \tag{24}$$

The **relative change matrix before clipping** is:

$$\mathbf{A}_\text{raw} = \Delta\mathbf{V}_\text{raw}\,\mathbf{V}^{-1} = \left(-\eta_v\,e_v\,\mathbf{S}\,\mathbf{V}\right)\mathbf{V}^{-1} = -\eta_v\,e_v\,\mathbf{S}\,\underbrace{\mathbf{V}\mathbf{V}^{-1}}_{=\,\mathbf{I}} \tag{25}$$

$$\mathbf{A}_\text{raw} = -\eta_v\,e_v\,\mathbf{S} \tag{26}$$

The $\mathbf{V}\mathbf{V}^{-1} = \mathbf{I}$ cancellation in (25) holds for any invertible $\mathbf{V}$ and does not require $\mathbf{R}_Z \approx \mathbf{I}$.

---

## 7. Incorporating the Trust-Region Clip

The global clip rescales $\Delta\mathbf{V}_\text{raw}$ so that the relative Frobenius-norm change does not exceed `max_rel_delta_v`:

$$\Delta\mathbf{V} = c\,\Delta\mathbf{V}_\text{raw}, \qquad c = \min\!\left(1,\;\frac{\texttt{max\_rel\_delta\_v}\cdot\|\mathbf{V}\|_F}{\|\Delta\mathbf{V}_\text{raw}\|_F}\right) \tag{27}$$

In the implementation $c$ is recovered from the norms of the clipped and unclipped steps:

$$c = \frac{\|\Delta\mathbf{V}\|_F}{\|\Delta\mathbf{V}_\text{raw}\|_F + \varepsilon} \tag{28}$$

The actual relative change matrix is:

$$\mathbf{A} = \Delta\mathbf{V}\,\mathbf{V}^{-1} = c\,\Delta\mathbf{V}_\text{raw}\,\mathbf{V}^{-1} = c\,\mathbf{A}_\text{raw} \overset{(26)}{=} -c\,\eta_v\,e_v\,\mathbf{S} \tag{29}$$

Substituting (29) into the coupling correction (21):

$$\Delta\mathbf{B}_\text{coupling} = -\mathbf{B}\,\mathbf{A} = -\mathbf{B}\left(-c\,\eta_v\,e_v\,\mathbf{S}\right) = c\,\eta_v\,e_v\,\mathbf{B}\mathbf{S} \tag{30}$$

Using the symmetry of $\mathbf{S}$ (equation 24), $\mathbf{S} = \mathbf{S}^\top$, so (30) can also be written as $\mathbf{B}\mathbf{S}^\top$, confirming the sign is positive.

---

## 8. The Coupling Matrix

Defining the **coupling matrix**:

$$\mathbf{\Gamma} = c\,\eta_v\,e_v\,\mathbf{S} \tag{31}$$

the correction takes the compact form:

$$\boxed{\Delta\mathbf{B}_\text{coupling} = \mathbf{B}\,\mathbf{\Gamma}} \tag{32}$$

All quantities in $\mathbf{\Gamma}$ are already computed during the V update:
- $\mathbf{S}$ is the `direction` variable (symmetric, $D \times D$)
- $e_v$ is the whitening error scalar
- $c$ is recovered from the two Frobenius norms (equation 28)
- $\eta_v$ is `config.eta_v`

No matrix inversion is required.

---

## 9. Application in the Algorithm

The correction (32) is applied to $\mathbf{B}$ **before** source estimates are formed, so that the subsequent spike detection and contrast update both operate in the corrected frame. With an optional trust-region clip at rate `max_rel_delta_b`:

$$\mathbf{B} \;\leftarrow\; \mathbf{B} + \text{clip}\!\left(\mathbf{B}\,\mathbf{\Gamma},\;\mathbf{B},\;\texttt{max\_rel\_delta\_b}\right) \tag{33}$$

followed by QR orthonormalisation (which the contrast update in `update_B_spike_gated` already applies at the end of the same batch).

---

## 10. Validity Conditions and Approximation Error

The derivation rests on two conditions:

**C1 — First-order approximation (equation 18).**  
The error is $O(\|\mathbf{A}\|^2) = O((c\,\eta_v\,|e_v|\,\|\mathbf{S}\|)^2)$. The clip (27) bounds $c\,\|\Delta\mathbf{V}_\text{raw}\|_F \leq \texttt{max\_rel\_delta\_v}\cdot\|\mathbf{V}\|_F$, which is typically $\leq 10^{-1}$, making the quadratic error at most $\sim 1\%$ of the linear term.

**C2 — Full column rank of $\mathbf{Z}$ (equation 14→15).**  
The FIFO maintains $N_\text{fifo} \geq D$ samples and shrinkage regularisation ensures $\mathbf{R}_Z \succ 0$, so $\mathbf{Z}$ has rank $D$ with probability one.

**Note on `wh_trace_renorm`.**  
If `wh_trace_renorm=True`, a scalar rescaling $\mathbf{V} \leftarrow \sqrt{\tau}\,\mathbf{V}$ is applied after the natural-gradient step. This rescaling contributes a term $(\sqrt{\tau}-1)\mathbf{I}$ to $\mathbf{A}$, which is not captured by (31). The induced coupling correction would be $(\sqrt{\tau}-1)\mathbf{B}$, i.e. a uniform scaling of all rows of $\mathbf{B}$. Since QR orthonormalisation normalises the rows, this term is projected out automatically and can be ignored in practice.

---

## Summary

| Step | Equation | Key result |
|------|----------|------------|
| Forward model | (1)–(3) | $\mathbf{Y} = \mathbf{X}\mathbf{V}^\top\mathbf{B}^\top$ |
| V update effect | (4)–(11) | $\mathbf{Z}' = \mathbf{Z}(\mathbf{I} + \mathbf{A}^\top)$, $\mathbf{A} = \Delta\mathbf{V}\mathbf{V}^{-1}$ |
| Consistency | (12)–(17) | $\mathbf{B}' = \mathbf{B}(\mathbf{I}+\mathbf{A})^{-1}$ (exact) |
| First-order expansion | (18)–(21) | $\Delta\mathbf{B}_\text{coupling} \approx -\mathbf{B}\mathbf{A}$ |
| Natural gradient cancels $\mathbf{V}^{-1}$ | (22)–(26) | $\mathbf{A}_\text{raw} = -\eta_v e_v \mathbf{S}$ |
| Clip scale | (27)–(29) | $\mathbf{A} = -c\,\eta_v e_v \mathbf{S}$ |
| Final correction | (30)–(32) | $\Delta\mathbf{B}_\text{coupling} = \mathbf{B}\,\mathbf{\Gamma}$, $\mathbf{\Gamma} = c\,\eta_v e_v \mathbf{S}$ |
