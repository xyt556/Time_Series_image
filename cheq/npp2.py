import numpy as np
from osgeo import gdal
from scipy.stats import rankdata
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def write_geotiff(filename, data, reference_file):
    dataset = gdal.Open(reference_file, gdal.GA_ReadOnly)
    driver = gdal.GetDriverByName('GTiff')
    output_dataset = driver.Create(filename, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
    output_dataset.SetProjection(dataset.GetProjection())
    output_dataset.SetGeoTransform(dataset.GetGeoTransform())
    band = output_dataset.GetRasterBand(1)
    band.WriteArray(data)
    output_dataset = None

# 打开多年的TIFF文件并提取NPP数据
# image_files = ['CHEQ_500M_2005.tif', 'CHEQ_500M_2010.tif', 'CHEQ_500M_2015.tif', 'CHEQ_500M_2020.tif']  # 替换成你的TIFF文件名列表

import os
# *********************************获取指定目录下的文件名***********************************
# 指定目录路径
directory_path = './NPP'

# 初始化一个空列表，用于存储.tif文件名
image_files = []

# 遍历指定目录中的文件
for filename in os.listdir(directory_path):
    if filename.endswith('.tif'):
        image_files.append(directory_path + '/'+ filename)

# 打印存储的.tif文件名列表
print(image_files)
# *********************************获取指定目录下的文件名***********************************

npp_data = []

for image_file in image_files:
    dataset = gdal.Open(image_file, gdal.GA_ReadOnly)
    npp_array = dataset.ReadAsArray()
    # 替换-1为NaN
    # npp_array = np.where(npp_array == -1, np.nan, npp_array)
    npp_data.append(npp_array)

# 将多年数据构建为时间序列
time_series = np.array(npp_data)

# Mann-Kendall趋势检验
def mann_kendall_trend(data):
    n = len(data)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(data[j] - data[i])
    var_s = (n * (n - 1) * (2 * n + 5)) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return z, p_value

# 计算每像素的Mann-Kendall检验p值
z = np.zeros_like(npp_data[0], dtype=float)
p = np.zeros_like(npp_data[0], dtype=float)
for i in range(npp_data[0].shape[0]):
    for j in range(npp_data[0].shape[1]):
        pixel_values = time_series[:, i, j]
        # 移除NaN值
        valid_pixel_values = pixel_values[~np.isnan(pixel_values)]
        if len(valid_pixel_values) > 0:
            z[i, j], p[i, j] = mann_kendall_trend(valid_pixel_values)

# Sen's Slope方法
def sen_slope(data):
    n = len(data)
    slope_values = []

    for m in range(n - 1):
        for k in range(m + 1, n):
            delta_npp = data[k] - data[m]
            delta_years = k - m
            slope = delta_npp / delta_years
            slope_values.append(slope)

    sen_slope_value = np.median(np.sort(slope_values))
    return sen_slope_value

sen_slopes = np.zeros_like(npp_data[0], dtype=float)
for i in range(npp_data[0].shape[0]):
    for j in range(npp_data[0].shape[1]):
        pixel_values = time_series[:, i, j]
        # 移除NaN值
        valid_pixel_values = pixel_values[~np.isnan(pixel_values)]
        if len(valid_pixel_values) > 0:
            sen_slopes[i, j] = sen_slope(valid_pixel_values)

# 保存Mann-Kendall检验的p值为TIFF文件
output_z_values_file = 'mann_kendall_z_values.tif'
output_p_values_file = 'mann_kendall_p_values.tif'
write_geotiff(output_z_values_file, z, image_files[0])
write_geotiff(output_p_values_file, p, image_files[0])

# 保存Sen's Slope结果为TIFF文件
output_sen_slopes_file = "sen_slopes.tif"
write_geotiff(output_sen_slopes_file, sen_slopes, image_files[0])

# 可视化Mann-Kendall检验的p值
plt.imshow(p, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Mann-Kendall p-value')
plt.title('Mann-Kendall Test Results')
plt.show()
plt.imshow(z, cmap='coolwarm', interpolation='nearest')
plt.colorbar(label='Mann-Kendall z-value')
plt.title('Mann-Kendall Test Results')
plt.show()

# 可视化Sen's Slope结果
plt.imshow(sen_slopes, cmap='RdYlGn', interpolation='nearest')
plt.colorbar(label="Sen's Slope")
plt.title("Sen's Slope Results")
plt.show()
