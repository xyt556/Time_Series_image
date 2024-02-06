import numpy as np
from osgeo import gdal
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 打开两个栅格TIFF文件
tif_file1 = './NPP/NPP_2001.tif'
tif_file2 = './NPP/NPP_2022.tif'
dataset1 = gdal.Open(tif_file1, gdal.GA_ReadOnly)
dataset2 = gdal.Open(tif_file2, gdal.GA_ReadOnly)

# 读取栅格数据为NumPy数组
data1 = dataset1.ReadAsArray()
data2 = dataset2.ReadAsArray()
print(data1)
# 将特定的空值替换为NaN
data1 = np.where(data1 == 32767, np.nan, data1)
data2 = np.where(data2 == -1, np.nan, data2)

# 将数据展平为一维数组
data1_flat = data1.flatten()
data2_flat = data2.flatten()

# 排除NaN值
valid_indices = ~np.isnan(data1_flat) & ~np.isnan(data2_flat)
data1_flat = data1_flat[valid_indices]
data2_flat = data2_flat[valid_indices]

# 计算Pearson相关性
correlation_coefficient, _ = pearsonr(data1_flat, data2_flat)

# 输出相关性系数
print("Pearson相关性系数:", correlation_coefficient)

# 绘制散点图和拟合线
plt.figure(figsize=(8, 6))
plt.scatter(data1_flat, data2_flat, s=10, color='b', label='散点')
plt.xlabel('数据1')
plt.ylabel('数据2')

# 计算拟合线
fit = np.polyfit(data1_flat, data2_flat, 1)
fit_fn = np.poly1d(fit)

# 绘制拟合线
plt.plot(data1_flat, fit_fn(data1_flat), color='r', label='拟合线')

# 显示相关性系数在图上
plt.text(np.min(data1_flat), np.max(data2_flat), f'Pearson相关性系数: {correlation_coefficient:.2f}', fontsize=12, color='g')

plt.title('Pearson相关性分析')
plt.legend()
plt.grid()

# 绘制数据1的盒须图
plt.figure(figsize=(8, 6))
plt.boxplot(data1_flat, labels=['数据1'])
plt.title('数据1的盒须图')

# 绘制数据2的盒须图
plt.figure(figsize=(8, 6))
plt.boxplot(data2_flat, labels=['数据2'])
plt.title('数据2的盒须图')

plt.show()
