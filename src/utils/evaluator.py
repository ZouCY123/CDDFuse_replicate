"""
src/utils/evaluator.py — 图像融合评估指标

8 个指标全部是纯函数，输入均为 shape (H, W) 的 uint8 numpy 数组。

使用：
    from src.utils.evaluator import compute_all_metrics
    metrics = compute_all_metrics(fused, ir, vis)
"""

import numpy as np
import math
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as ssim
import sklearn.metrics as skm


def entropy(img: np.ndarray) -> float:
    """EN：信息熵，衡量图像信息量，越高越好"""
    a = np.uint8(np.round(img)).flatten()
    h = np.bincount(a, minlength=256) / a.shape[0]
    return -np.sum(h * np.log2(h + (h == 0)))


def std(img: np.ndarray) -> float:
    """SD：标准差，衡量图像对比度，越高越好"""
    return float(np.std(img))


def spatial_frequency(img: np.ndarray) -> float:
    """SF：空间频率，衡量图像细节丰富程度，越高越好"""
    rf = np.mean((img[:, 1:] - img[:, :-1]) ** 2)
    cf = np.mean((img[1:, :] - img[:-1, :]) ** 2)
    return float(np.sqrt(rf + cf))


def mutual_information(fused: np.ndarray, ir: np.ndarray, vis: np.ndarray) -> float:
    """MI：互信息，衡量融合图与源图像的信息相关性，越高越好"""
    mi_ir  = skm.mutual_info_score(fused.flatten(), ir.flatten())
    mi_vis = skm.mutual_info_score(fused.flatten(), vis.flatten())
    return float(mi_ir + mi_vis)


def scd(fused: np.ndarray, ir: np.ndarray, vis: np.ndarray) -> float:
    """SCD：差异相关性之和，越高越好"""
    def corr(a, b):
        a_c = a - a.mean()
        b_c = b - b.mean()
        denom = np.sqrt((a_c**2).sum() * (b_c**2).sum())
        return float(np.sum(a_c * b_c) / (denom + 1e-10))

    return corr(ir,  fused - vis) + corr(vis, fused - ir)


def _vif_single(ref: np.ndarray, dist: np.ndarray) -> float:
    """计算单对图像的 VIF"""
    sigma_nsq = 2
    eps       = 1e-10
    num, den  = 0.0, 0.0

    for scale in range(1, 5):
        N  = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0
        m, n = [(ss - 1.) / 2. for ss in (N, N)]
        y, x = np.ogrid[-m:m+1, -n:n+1]
        h    = np.exp(-(x*x + y*y) / (2. * sd * sd))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        win  = h / h.sum() if h.sum() != 0 else h

        if scale > 1:
            ref  = convolve2d(ref,  np.rot90(win, 2), mode='valid')[::2, ::2]
            dist = convolve2d(dist, np.rot90(win, 2), mode='valid')[::2, ::2]

        mu1    = convolve2d(ref,  np.rot90(win, 2), mode='valid')
        mu2    = convolve2d(dist, np.rot90(win, 2), mode='valid')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1mu2 = mu1 * mu2

        s1  = convolve2d(ref*ref,   np.rot90(win, 2), mode='valid') - mu1_sq
        s2  = convolve2d(dist*dist, np.rot90(win, 2), mode='valid') - mu2_sq
        s12 = convolve2d(ref*dist,  np.rot90(win, 2), mode='valid') - mu1mu2

        s1[s1 < 0] = 0
        s2[s2 < 0] = 0

        g    = s12 / (s1 + eps)
        sv_sq = s2 - g * s12

        g[s1 < eps]    = 0
        sv_sq[s1 < eps] = s2[s1 < eps]
        s1[s1 < eps]   = 0
        g[s2 < eps]    = 0
        sv_sq[s2 < eps] = 0
        sv_sq[g < 0]   = s2[g < 0]
        g[g < 0]       = 0
        sv_sq[sv_sq <= eps] = eps

        num += np.sum(np.log10(1 + g*g*s1 / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + s1 / sigma_nsq))

    vif = num / den if den > eps else 1.0
    return 1.0 if np.isnan(vif) else float(vif)


def vif(fused: np.ndarray, ir: np.ndarray, vis: np.ndarray) -> float:
    """VIF：视觉信息保真度，越高越好"""
    return _vif_single(ir, fused) + _vif_single(vis, fused)


def _qabf_array(img: np.ndarray):
    """计算 Qabf 所需的梯度幅值和方向"""
    h1 = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=np.float32)
    h3 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    sx = convolve2d(img, h3, mode='same')
    sy = convolve2d(img, h1, mode='same')
    g  = np.sqrt(sx**2 + sy**2)
    a  = np.where(sx == 0, math.pi/2, np.arctan(sy / (sx + 1e-10)))
    return g, a


def _qabf_score(ga, aa, gf, af):
    Tg, kg, Dg = 0.9994, -15, 0.5
    Ta, ka, Da = 0.9879, -22, 0.8
    GAF = np.where(ga > gf, gf/(ga+1e-10), np.where(ga == gf, gf, ga/(gf+1e-10)))
    AAF = 1 - np.abs(aa - af) / (math.pi / 2)
    return (Tg / (1 + np.exp(kg*(GAF-Dg)))) * (Ta / (1 + np.exp(ka*(AAF-Da))))


def qabf(fused: np.ndarray, ir: np.ndarray, vis: np.ndarray) -> float:
    """Qabf：基于梯度的融合质量，越高越好"""
    gA, aA = _qabf_array(ir)
    gB, aB = _qabf_array(vis)
    gF, aF = _qabf_array(fused)
    QAF = _qabf_score(gA, aA, gF, aF)
    QBF = _qabf_score(gB, aB, gF, aF)
    deno = np.sum(gA + gB) + 1e-10
    return float(np.sum(QAF*gA + QBF*gB) / deno)


def ssim_score(fused: np.ndarray, ir: np.ndarray, vis: np.ndarray) -> float:
    """SSIM：结构相似性之和，越高越好"""
    return float(ssim(fused, ir) + ssim(fused, vis))


# ── 统一入口 ──────────────────────────────────────────────────────

METRIC_NAMES = ['EN', 'SD', 'SF', 'MI', 'SCD', 'VIF', 'Qabf', 'SSIM']


def compute_all_metrics(
    fused: np.ndarray,
    ir:    np.ndarray,
    vis:   np.ndarray,
) -> dict[str, float]:
    """
    计算所有 8 个指标。

    Args:
        fused: 融合图像，shape (H, W)，uint8 或 float，值域 [0, 255]
        ir:    红外图像，shape (H, W)
        vis:   可见光图像，shape (H, W)

    Returns:
        {'EN': ..., 'SD': ..., 'SF': ..., 'MI': ...,
         'SCD': ..., 'VIF': ..., 'Qabf': ..., 'SSIM': ...}
    """
    return {
        'EN':   entropy(fused),
        'SD':   std(fused),
        'SF':   spatial_frequency(fused),
        'MI':   mutual_information(fused, ir, vis),
        'SCD':  scd(fused, ir, vis),
        'VIF':  vif(fused, ir, vis),
        'Qabf': qabf(fused, ir, vis),
        'SSIM': ssim_score(fused, ir, vis),
    }