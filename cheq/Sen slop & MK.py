import numpy as np
from osgeo import gdal
from scipy.stats import rankdata  # 导入 rankdata 函数
from scipy.stats import norm

# 输入多年NPP影像数据，假设数据已按年份排列，且为TIF格式
image_files = ['CHEQ_500M_2005.tif', 'CHEQ_500M_2010.tif', 'CHEQ_500M_2015.tif', 'CHEQ_500M_2020.tif']  # 替换成你的影像文件名列表

# 存储提取的像素值
npp_data = []

# 图像处理和数据提取
for image_file in image_files:
    dataset = gdal.Open(image_file, gdal.GA_ReadOnly)
    npp_data.append(dataset.ReadAsArray())

# 创建GDAL驱动程序
driver = gdal.GetDriverByName('GTiff')

# 创建一个与影像尺寸相同的空数组，用于存储Sen's Slope结果
sen_slopes = np.zeros_like(npp_data[0], dtype=float)

# 创建一个与影像尺寸相同的空数组，用于存储Mann-Kendall检验的p值
p_values = np.zeros_like(npp_data[0], dtype=float)

# Mann-Kendall趋势检验
for i in range(npp_data[0].shape[0]):
    for j in range(npp_data[0].shape[1]):
        # 提取每年的像素值
        pixel_values = [npp[i, j] for npp in npp_data]

        # 计算Sen's Slope
        n = len(pixel_values)
        slope_values = []

        for m in range(n - 1):
            for k in range(m + 1, n):
                delta_npp = pixel_values[k] - pixel_values[m]
                delta_years = k - m
                slope = delta_npp / delta_years
                slope_values.append(slope)

        # 计算Sen's Slope（中位数）
        sen_slope = np.median(np.sort(slope_values))

        # 存储Sen's Slope结果
        sen_slopes[i, j] = sen_slope

        # Mann-Kendall趋势检验
        ranked_data = rankdata(pixel_values)
        n_ranked = len(ranked_data)
        S = 0
        for p in range(n_ranked - 1):
            for q in range(p + 1, n_ranked):
                S += np.sign(ranked_data[q] - ranked_data[p])
        var_S = (n_ranked * (n_ranked - 1) * (2 * n_ranked + 5)) / 18
        Z = (S - 1) / np.sqrt(var_S)
        p_value = 2 * (1 - norm.cdf(abs(Z)))  # 使用 norm 函数计算 p-value

        # 存储Mann-Kendall检验的p值
        p_values[i, j] = p_value

# 输出Sen's Slope和Mann-Kendall检验结果

# 定义输出结果的TIF文件路径
output_file = 'npp_trend_result.tif'

# 获取原始影像的地理空间信息
original_dataset = gdal.Open(image_files[0], gdal.GA_ReadOnly)
geotransform = original_dataset.GetGeoTransform()
projection = original_dataset.GetProjection()

# 创建输出TIF文件
output_dataset = driver.Create(output_file, npp_data[0].shape[1], npp_data[0].shape[0], 2, gdal.GDT_Float32)

# 写入Sen's Slope结果
output_dataset.GetRasterBand(1).WriteArray(sen_slopes)

# 写入Mann-Kendall检验的p值
output_dataset.GetRasterBand(2).WriteArray(p_values)

# 设置地理空间信息
output_dataset.SetGeoTransform(geotransform)
output_dataset.SetProjection(projection)

# 保存和关闭数据集
output_dataset.FlushCache()
output_dataset = None

# 这个示例程序演示了如何执行Sen's Slope方法和Mann-Kendall趋势检验，并将结果写出为TIF文件。
