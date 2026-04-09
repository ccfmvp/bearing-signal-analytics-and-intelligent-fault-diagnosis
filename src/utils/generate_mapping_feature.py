import pandas as pd

# 创建特征数据
features_data = {
    '索引': list(range(55)),
    '特征名称': [
        'mean', 'median', 'std', 'variance', 'rms', 'peak', 'peak_to_peak', 'skewness',
        'kurtosis', 'crest_factor', 'form_factor', 'impulse_factor', 'energy', 'mean_abs',
        'abs_energy', 'q25', 'q75', 'iqr', 'zero_crossing_rate', 'peak_count', 'peak_rate',
        'autocorr_max', 'autocorr_first_zero', 'shannon_entropy', 'envelope_mean',
        'envelope_std', 'diff_mean', 'diff_std', 'fractal_dimension', 'lyapunov_exponent',
        'spectral_centroid', 'spectral_spread', 'spectral_skewness', 'spectral_kurtosis',
        'spectral_energy', 'spectral_entropy', 'band_1_energy', 'band_2_energy',
        'band_3_energy', 'band_4_energy', 'band_5_energy', 'band_1_energy_ratio',
        'band_2_energy_ratio', 'band_3_energy_ratio', 'band_4_energy_ratio',
        'band_5_energy_ratio', 'dominant_frequency', 'dominant_magnitude',
        'dominant_frequency_ratio', 'frequency_std', 'frequency_variance',
        'spectral_rolloff', 'spectral_flatness', 'spectrogram_peak_var', 'spectrogram_entropy'
    ],
    '描述': [
        '均值', '中位数', '标准差', '方差', '均方根值', '峰值', '峰峰值', '偏度',
        '峰度', '峰值因子', '波形因子', '脉冲因子', '能量', '平均绝对值',
        '绝对能量', '25%分位数', '75%分位数', '四分位距', '过零率', '峰值数量',
        '峰值率', '自相关最大值', '自相关第一个过零点时间', '香农熵', '包络均值',
        '包络标准差', '差分均值', '差分标准差', '分形维度', '李雅普诺夫指数',
        '频谱重心', '频谱扩展', '频谱偏度', '频谱峰度', '频谱能量', '频谱熵',
        '频带1能量', '频带2能量', '频带3能量', '频带4能量', '频带5能量',
        '频带1能量占比', '频带2能量占比', '频带3能量占比', '频带4能量占比',
        '频带5能量占比', '主导频率', '主导频率幅值', '主导频率占比', '频率标准差',
        '频率方差', '频谱滚降点', '频谱平坦度', '时频谱峰值方差', '时频谱熵'
    ]
}

# 创建DataFrame
df = pd.DataFrame(features_data)

# 保存为CSV文件
df.to_csv('mappingFeature.csv', index=False, encoding='utf-8-sig')

print("特征表已成功保存为 'mappingFeature.csv'")
print(f"共保存了 {len(df)} 个特征")