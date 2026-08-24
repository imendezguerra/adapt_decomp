"""PCA dimensionality reduction for extended EMG data.

Called only from cbss/core.py at calibration time -- online adaptation
(adaptation.py/data_structures.py) never calls this function again,
it only consumes the fitted pca_components/pca_mean outputs stored on
CBSSResult and threaded through Decomposition._apply_pca.
"""

import torch
from sklearn.decomposition import PCA
from typing import Optional, Tuple

def pca_reduction(
    emg_ext: torch.Tensor,
    n_components: Optional[int],
    pca_model: Optional[PCA] = None,
) -> Tuple[torch.Tensor, Optional[PCA]]:
    """Performs PCA dimensionality reduction (optional after extension).

    Args:
        emg_ext (torch.Tensor): Extended EMG data of shape (samples, channels).
        n_components (Optional[int]): Number of principal components to keep.
        pca_model (Optional[PCA], optional): Fitted PCA model. Defaults to None.

    Raises:
        ValueError: If the number of components is invalid.

    Returns:
        Tuple[torch.Tensor, Optional[PCA]]: Reduced EMG data and the fitted PCA model.
    """

    if n_components is None:
        return emg_ext, None
    d = min(n_components, emg_ext.shape[0], emg_ext.shape[1])
    if d < 1:
        raise ValueError("PCA requires at least one sample and one feature.")
    emg_ext_np = emg_ext.cpu().numpy()
    if pca_model is None:
        pca_model = PCA(n_components=d)
        emg_pca_np = pca_model.fit_transform(emg_ext_np)
    else:
        emg_pca_np = pca_model.transform(emg_ext_np)
    emg_pca = torch.from_numpy(emg_pca_np).to(device=emg_ext.device, dtype=emg_ext.dtype)
    return emg_pca, pca_model
