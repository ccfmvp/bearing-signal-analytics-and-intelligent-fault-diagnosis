# %%
import scipy.io
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

# %%
expanded_df = pd.read_excel('周期计算.xlsx', index_col=None)

# %%
import numpy as np
from scipy import signal, stats, fft
import matplotlib.pyplot as plt


def fractal_dimension(signal):
    """计算分形维度，使用盒子计数法"""
    N = len(signal)
    boxes = range(1, N // 2)
    counts = []

    for box_size in boxes:
        count = 0
        for start in range(0, N - box_size, box_size):
            segment = signal[start:start + box_size]
            if np.max(segment) > np.min(segment):  # 只考虑有效的段
                count += 1
        counts.append(count)

    log_counts = np.log(counts)
    log_boxes = np.log(boxes)

    # 计算斜率（即分形维度）
    coeffs = np.polyfit(log_boxes, log_counts, 1)
    return -coeffs[0]  # 分形维度是斜率的负值


def lyapunov_exponent(signal):
    """计算李雅普诺夫指数"""
    n = len(signal)
    delta_t = 1  # 时间步长
    n_neighbors = 5  # 邻居点的个数
    distances = np.zeros(n_neighbors)

    # 计算相邻点的距离
    for i in range(1, n - delta_t):
        dist = np.abs(signal[i + delta_t] - signal[i])
        distances = np.append(distances, dist)

    # 计算李雅普诺夫指数
    return np.mean(np.log(distances / delta_t))

def extract_time_frequency_features(signal_data, fs, feature_type='both'):
    """
    从滑动窗口采样后的振动信号中提取时域和频域特征
    
    参数:
    signal_data: 输入信号数据（一维数组）
    fs: 采样频率（Hz）
    feature_type: 提取的特征类型，'time'=时域, 'frequency'=频域, 'both'=两者都提取
    
    返回:
    features: 包含所有特征的字典
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
        features['rms'] = np.sqrt(np.mean(signal_data**2))  # 均方根值
        features['peak'] = np.max(np.abs(signal_data))  # 峰值
        features['peak_to_peak'] = np.ptp(signal_data)  # 峰峰值
        
        # 2. 形状特征
        features['skewness'] = stats.skew(signal_data)  # 偏度
        features['kurtosis'] = stats.kurtosis(signal_data)  # 峰度
        features['crest_factor'] = features['peak'] / features['rms'] if features['rms'] != 0 else 0  # 峰值因子
        features['form_factor'] = features['rms'] / np.mean(np.abs(signal_data)) if np.mean(np.abs(signal_data)) != 0 else 0  # 波形因子
        features['impulse_factor'] = features['peak'] / np.mean(np.abs(signal_data)) if np.mean(np.abs(signal_data)) != 0 else 0  # 脉冲因子
        
        # 3. 高级统计特征
        features['energy'] = np.sum(signal_data**2)  # 能量
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
        peaks, _ = signal.find_peaks(np.abs(signal_data), distance=fs//10)  # 避免过于密集的峰值
        features['peak_count'] = len(peaks)  # 峰值数量
        features['peak_rate'] = len(peaks) / t_total if t_total > 0 else 0  # 峰值率
        
        # 6. 自相关特征
        autocorr = np.correlate(signal_data, signal_data, mode='full')
        autocorr = autocorr[autocorr.size//2:]  # 取一半
        features['autocorr_max'] = np.max(autocorr)  # 自相关最大值
        features['autocorr_first_zero'] = np.argmin(np.abs(autocorr[:fs])) / fs if np.any(autocorr[:fs] < 0) else 0  # 第一个过零点时间
        
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

        #10. 计算分形维度和李雅普诺夫指数
        features['fractal_dimension'] = fractal_dimension(signal_data)
        features['lyapunov_exponent'] = lyapunov_exponent(signal_data)
    
    if feature_type in ['frequency', 'both']:
        # ==================== 频域特征 ====================
        # 1. 傅里叶变换获取频谱
        fft_values = fft.fft(signal_data)
        fft_magnitude = np.abs(fft_values[:n//2])  # 取一半（对称性）
        freqs = fft.fftfreq(n, 1/fs)[:n//2]
        
        # 排除直流分量
        fft_magnitude = fft_magnitude[1:]
        freqs = freqs[1:]
        
        # 2. 频谱统计特征
        features['spectral_centroid'] = np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude) if np.sum(fft_magnitude) > 0 else 0  # 频谱重心
        features['spectral_spread'] = np.sqrt(np.sum((freqs - features['spectral_centroid'])**2 * fft_magnitude) / np.sum(fft_magnitude)) if np.sum(fft_magnitude) > 0 else 0  # 频谱扩展
        features['spectral_skewness'] = np.sum((freqs - features['spectral_centroid'])**3 * fft_magnitude) / (features['spectral_spread']**3 * np.sum(fft_magnitude)) if features['spectral_spread'] > 0 and np.sum(fft_magnitude) > 0 else 0  # 频谱偏度
        features['spectral_kurtosis'] = np.sum((freqs - features['spectral_centroid'])**4 * fft_magnitude) / (features['spectral_spread']**4 * np.sum(fft_magnitude)) if features['spectral_spread'] > 0 and np.sum(fft_magnitude) > 0 else 0  # 频谱峰度
        
        # 3. 频谱能量特征
        total_power = np.sum(fft_magnitude**2)
        features['spectral_energy'] = total_power  # 频谱能量
        features['spectral_entropy'] = -np.sum((fft_magnitude**2 / total_power) * np.log2(fft_magnitude**2 / total_power)) if total_power > 0 else 0  # 频谱熵
        
        # 4. 频带能量特征 (划分为5个频带)
        band_edges = np.linspace(0, fs/2, 6)  # 将0-fs/2分为5个频带
        band_energy = []
        for i in range(len(band_edges)-1):
            mask = (freqs >= band_edges[i]) & (freqs < band_edges[i+1])
            band_power = np.sum(fft_magnitude[mask]**2)
            band_energy.append(band_power)
            features[f'band_{i+1}_energy'] = band_power
            features[f'band_{i+1}_energy_ratio'] = band_power / total_power if total_power > 0 else 0
        
        # 5. 主导频率特征
        dominant_freq_idx = np.argmax(fft_magnitude)
        features['dominant_frequency'] = freqs[dominant_freq_idx]  # 主导频率
        features['dominant_magnitude'] = fft_magnitude[dominant_freq_idx]  # 主导频率幅值
        features['dominant_frequency_ratio'] = features['dominant_magnitude'] / np.sum(fft_magnitude) if np.sum(fft_magnitude) > 0 else 0  # 主导频率占比
        
        # 6. 频率标准差和方差
        features['frequency_std'] = np.sqrt(np.sum(fft_magnitude**2 * (freqs - features['spectral_centroid'])**2) / total_power) if total_power > 0 else 0
        features['frequency_variance'] = features['frequency_std']**2
        
        # 7. 频谱滚降点 (95%能量点)
        cumulative_energy = np.cumsum(fft_magnitude**2)
        rolloff_point = np.where(cumulative_energy >= 0.95 * total_power)[0]
        features['spectral_rolloff'] = freqs[rolloff_point[0]] if len(rolloff_point) > 0 else 0
        
        # 8. 频谱平坦度
        features['spectral_flatness'] = np.exp(np.mean(np.log(fft_magnitude + 1e-12))) / np.mean(fft_magnitude) if np.mean(fft_magnitude) > 0 else 0
        
        # 9. 短时傅里叶变换特征 (时频分析)
        f, t, Sxx = signal.spectrogram(signal_data, fs, nperseg=min(256, n//8))
        features['spectrogram_peak_var'] = np.var(np.max(Sxx, axis=0))  # 时频谱峰值方差
        features['spectrogram_entropy'] = -np.sum(Sxx / np.sum(Sxx) * np.log2(Sxx / np.sum(Sxx) + 1e-12))  # 时频谱熵
    
    return features

# %%
for index, row in tqdm(expanded_df.iterrows(), total=expanded_df.shape[0]):
    all_features = []
    df = pd.read_csv(f'源域数据集滑动窗口采样结果/{index+1:03d}.csv', index_col=None, header=0)
    fs = 48000 if row['Level_2'].startswith('48') else 12000
    file_features = []
    for i in range(df.shape[0]):
        features = extract_time_frequency_features(np.array(df.iloc[i, :]), fs, 'both')
        file_features.append(list(features.values()))
    all_features.append(np.array(file_features))
    all_features = np.array(all_features)
    all_features = all_features[0, :, :]
    all_features = pd.DataFrame(all_features)
    all_features.to_csv(f'test/特征提取_{index+1:03d}.csv', index=None)

# %%
# for index, row in tqdm(expanded_df.iterrows(), total=expanded_df.shape[0]):
#     all_features = []
#     if row['Level_2'] == '48kHz_Normal_data':
#         df = pd.read_csv(f'源域数据集滑动窗口采样结果/{index+1:03d}.csv', index_col=None, header=0)
#         fs = 48000 if row['Level_2'].startswith('48') else 12000
#         file_features = []
#         for i in range(df.shape[0]):
#             features = extract_time_frequency_features(np.array(df.iloc[i, :]), fs, 'both')
#             file_features.append(list(features.values()))
#         all_features.append(np.array(file_features))
#         all_features = np.array(all_features)
#         all_features = all_features[0, :, :]
#         all_features = pd.DataFrame(all_features)
#         all_features.to_csv(f'test/特征提取_{index+1:03d}.csv', index=None)


