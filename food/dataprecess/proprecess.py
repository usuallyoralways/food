from boxsers.preprocessing import savgol_smoothing, als_baseline_cor, spectral_normalization
from boxsers.visual_tools import spectro_plot


# savgol_smoothing: Savitzky-Golay滤波器 ,曲线光滑

# spectral_normalization 光谱正则化

# 从csv 文件读入数据，其中第一行或者第一列为标签数据，如果row=1, 则每一列代表一个拉曼光谱，若row=0，则每一行为一个拉曼光谱。
def read_csv(row=1):
    pass