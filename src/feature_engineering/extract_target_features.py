import scipy.io
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

import numpy as np
from scipy import signal, stats, fft
import matplotlib.pyplot as plt

def fractal_dimension(signal):
    N = len(signal)
    # 选择适当的盒子大小范围（对数尺度）
    box_sizes = np.logspace(0.5, np.log10(N // 4), num=10, dtype=int)

    if len(box_sizes) < 3:
        return 1.0  # 盒子大小种类太少

    counts = []
    valid_boxes = []

    signal_range = np.max(signal) - np.min(signal)
    if signal_range == 0:
        return 1.0  # 常数值信号

    # 归一化信号到[0,1]范围
    normalized_signal = (signal - np.min(signal)) / signal_range

    for box_size in box_sizes:
        # 计算在时间方向上的盒子数
        time_boxes = N // box_size
        if time_boxes < 1:
            continue

        # 计算每个时间盒子中的值范围，确定需要的垂直盒子数
        total_boxes = 0
        for i in range(0, N - box_size + 1, box_size):
            segment = normalized_signal[i:i + box_size]
            if len(segment) > 0:
                min_val = np.min(segment)
                max_val = np.max(segment)
                # 垂直方向需要的盒子数（按比例缩放）
                vertical_boxes = max(1, int((max_val - min_val) * box_size) + 1)
                total_boxes += vertical_boxes

        if total_boxes > 0:
            counts.append(total_boxes)
            valid_boxes.append(box_size)

    if len(counts) < 3:
        return 1.0  # 有效数据点不足

    # 对数变换和线性拟合
    log_boxes = np.log(valid_boxes)
    log_counts = np.log(counts)

    try:
        # 使用加权最小二乘法，更稳定
        coeffs = np.polyfit(log_boxes, log_counts, 1)
        fractal_dim = -coeffs[0]
        # 确保分形维度在合理范围内
        return max(1.0, min(2.0, fractal_dim))
    except:
        return 1.0  # 拟合失败返回默认值

def lyapunov_exponent(signal):
    """计算李雅普诺夫指数 - 稳健版本"""
    if len(signal) < 20:
        return 0.0  # 信号太短

    n = len(signal)
    delta_t = 1

    # 方法1：使用相邻点距离变化（更稳定）
    distances = []
    epsilon = 1e-10  # 小常数避免log(0)

    for i in range(n - delta_t):
        current_point = signal[i]
        next_point = signal[i + delta_t]

        # 计算距离（绝对值差）
        distance = abs(next_point - current_point)

        if distance > epsilon:  # 只考虑有意义的距离
            # 计算指数增长率
            growth_rate = np.log(distance + epsilon)
            distances.append(growth_rate)

    if len(distances) == 0:
        return 0.0  # 没有有效距离数据

    # 使用trimmed mean提高鲁棒性（去除极端值）
    distances_array = np.array(distances)
    q25, q75 = np.percentile(distances_array, [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr

    valid_distances = distances_array[(distances_array >= lower_bound) & (distances_array <= upper_bound)]

    if len(valid_distances) == 0:
        return 0.0

    lyap_exp = np.mean(valid_distances)

    # 确保结果在合理范围内
    if np.isinf(lyap_exp) or np.isnan(lyap_exp):
        return 0.0

    return float(lyap_exp)

def extract_time_frequency_features(signal_data, fs, feature_type='both'):
    """
    从滑动窗口采样后的振动信号中提取时域和频域特征
    """
    features = {}

    # 基本信号信息
    n = len(signal_data)
    t_total = n / fs  # 总时间长度

    if feature_type in ['time', 'both']:
        # ==================== 时域特征 ====================
        # 1. 基本统计特征
        features['mean'] = np.mean(signal_data)
        features['median'] = np.median(signal_data)
        features['std'] = np.std(signal_data)
        features['variance'] = np.var(signal_data)
        features['rms'] = np.sqrt(np.mean(signal_data ** 2))  # 均方根值
        features['peak'] = np.max(np.abs(signal_data))  # 峰值
        features['peak_to_peak'] = np.ptp(signal_data)  # 峰峰值

        # 2. 形状特征
        features['skewness'] = stats.skew(signal_data)  # 偏度
        features['kurtosis'] = stats.kurtosis(signal_data)  # 峰度
        features['crest_factor'] = features['peak'] / features['rms'] if features['rms'] != 0 else 0  # 峰值因子
        features['form_factor'] = features['rms'] / np.mean(np.abs(signal_data)) if np.mean(
            np.abs(signal_data)) != 0 else 0  # 波形因子
        features['impulse_factor'] = features['peak'] / np.mean(np.abs(signal_data)) if np.mean(
            np.abs(signal_data)) != 0 else 0  # 脉冲因子

        # 3. 高级统计特征
        features['energy'] = np.sum(signal_data ** 2)  # 能量
        features['mean_abs'] = np.mean(np.abs(signal_data))  # 平均绝对值
        features['abs_energy'] = np.sum(np.abs(signal_data))  # 绝对能量

        # 4. 基于分位数的特征
        features['q25'] = np.percentile(signal_data, 25)
        features['q75'] = np.percentile(signal_data, 75)
        features['iqr'] = features['q75'] - features['q25']  # 四分位距

        # 5. 过零率与峰值计数
        zero_crossings = np.where(np.diff(np.sign(signal_data)))[0]
        features['zero_crossing_rate'] = len(zero_crossings) / n  # 过零率

        # 查找峰值
        peaks, _ = signal.find_peaks(np.abs(signal_data), distance=fs // 10)  # 避免过于密集的峰值
        features['peak_count'] = len(peaks)  # 峰值数量
        features['peak_rate'] = len(peaks) / t_total if t_total > 0 else 0  # 峰值率

        # 6. 自相关特征
        autocorr = np.correlate(signal_data, signal_data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]  # 取一半
        features['autocorr_max'] = np.max(autocorr)  # 自相关最大值
        features['autocorr_first_zero'] = np.argmin(np.abs(autocorr[:fs])) / fs if np.any(
            autocorr[:fs] < 0) else 0  # 第一个过零点时间

        # 7. 信号熵特征
        histogram, _ = np.histogram(signal_data, bins=50, density=True)
        histogram = histogram[histogram > 0]
        features['shannon_entropy'] = -np.sum(histogram * np.log2(histogram))  # 香农熵

        # 8. 时域包络特征
        analytic_signal = signal.hilbert(signal_data)
        amplitude_envelope = np.abs(analytic_signal)
        features['envelope_mean'] = np.mean(amplitude_envelope)
        features['envelope_std'] = np.std(amplitude_envelope)

        # 9. 差分特征
        diff_signal = np.diff(signal_data)
        features['diff_mean'] = np.mean(diff_signal)
        features['diff_std'] = np.std(diff_signal)

        # 10. 计算分形维度和李雅普诺夫指数
        features['fractal_dimension'] = fractal_dimension(signal_data)
        features['lyapunov_exponent'] = lyapunov_exponent(signal_data)

    if feature_type in ['frequency', 'both']:
        # ==================== 频域特征 ====================
        # 1. 傅里叶变换获取频谱
        fft_values = fft.fft(signal_data)
        fft_magnitude = np.abs(fft_values[:n // 2])  # 取一半（对称性）
        freqs = fft.fftfreq(n, 1 / fs)[:n // 2]

        # 排除直流分量
        fft_magnitude = fft_magnitude[1:]
        freqs = freqs[1:]

        # 2. 频谱统计特征
        features['spectral_centroid'] = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude) if np.sum(
            fft_magnitude) > 0 else 0  # 频谱重心
        features['spectral_spread'] = np.sqrt(
            np.sum((freqs - features['spectral_centroid']) ** 2 * fft_magnitude) / np.sum(fft_magnitude)) if np.sum(
            fft_magnitude) > 0 else 0  # 频谱扩展
        features['spectral_skewness'] = np.sum((freqs - features['spectral_centroid']) ** 3 * fft_magnitude) / (
                    features['spectral_spread'] ** 3 * np.sum(fft_magnitude)) if features[
                                                                                     'spectral_spread'] > 0 and np.sum(
            fft_magnitude) > 0 else 0  # 频谱偏度
        features['spectral_kurtosis'] = np.sum((freqs - features['spectral_centroid']) ** 4 * fft_magnitude) / (
                    features['spectral_spread'] ** 4 * np.sum(fft_magnitude)) if features[
                                                                                     'spectral_spread'] > 0 and np.sum(
            fft_magnitude) > 0 else 0  # 频谱峰度

        # 3. 频谱能量特征
        total_power = np.sum(fft_magnitude ** 2)
        features['spectral_energy'] = total_power  # 频谱能量
        features['spectral_entropy'] = -np.sum((fft_magnitude ** 2 / total_power) * np.log2(
            fft_magnitude ** 2 / total_power)) if total_power > 0 else 0  # 频谱熵

        # 4. 频带能量特征 (划分为5个频带)
        band_edges = np.linspace(0, fs / 2, 6)  # 将0-fs/2分为5个频带
        band_energy = []
        for i in range(len(band_edges) - 1):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
            band_power = np.sum(fft_magnitude[mask] ** 2)
            band_energy.append(band_power)
            features[f'band_{i + 1}_energy'] = band_power
            features[f'band_{i + 1}_energy_ratio'] = band_power / total_power if total_power > 0 else 0

        # 5. 主导频率特征
        dominant_freq_idx = np.argmax(fft_magnitude)
        features['dominant_frequency'] = freqs[dominant_freq_idx]  # 主导频率
        features['dominant_magnitude'] = fft_magnitude[dominant_freq_idx]  # 主导频率幅值
        features['dominant_frequency_ratio'] = features['dominant_magnitude'] / np.sum(fft_magnitude) if np.sum(
            fft_magnitude) > 0 else 0  # 主导频率占比

        # 6. 频率标准差和方差
        features['frequency_std'] = np.sqrt(np.sum(
            fft_magnitude ** 2 * (freqs - features['spectral_centroid']) ** 2) / total_power) if total_power > 0 else 0
        features['frequency_variance'] = features['frequency_std'] ** 2

        # 7. 频谱滚降点 (95%能量点)
        cumulative_energy = np.cumsum(fft_magnitude ** 2)
        rolloff_point = np.where(cumulative_energy >= 0.95 * total_power)[0]
        features['spectral_rolloff'] = freqs[rolloff_point[0]] if len(rolloff_point) > 0 else 0

        # 8. 频谱平坦度
        features['spectral_flatness'] = np.exp(np.mean(np.log(fft_magnitude + 1e-12))) / np.mean(
            fft_magnitude) if np.mean(fft_magnitude) > 0 else 0

        # 9. 短时傅里叶变换特征 (时频分析)
        f, t, Sxx = signal.spectrogram(signal_data, fs, nperseg=min(256, n // 8))
        features['spectrogram_peak_var'] = np.var(np.max(Sxx, axis=0))  # 时频谱峰值方差
        features['spectrogram_entropy'] = -np.sum(Sxx / np.sum(Sxx) * np.log2(Sxx / np.sum(Sxx) + 1e-12))  # 时频谱熵

    return features

for ID in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']:
    all_features = []
    df = pd.read_csv(f'target_sliding_window_result/{ID}.csv', index_col=None, header=0)
    fs = 32000
    file_features = []
    for i in range(df.shape[0]):
        features = extract_time_frequency_features(np.array(df.iloc[i, :]), fs, 'both')
        file_features.append(list(features.values()))
    all_features.append(np.array(file_features))
    all_features = np.array(all_features)
    all_features = all_features[0, :, :]
    all_features = pd.DataFrame(all_features)
    all_features.to_csv(f'target_domain_features/目标域特征提取_{ID}.csv', index=None)