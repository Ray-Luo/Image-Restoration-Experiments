import scipy.stats as stats
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



# ********************************************* EDSR *********************************************
if 1:
    navie_psnr_rgb= [ 55.2193, 58.5961, 41.0566, 40.673, 58.2001, 48.41, 42.2467, 40.0702, 55.6785, 48.9122, 48.7195, 43.3138, 39.6008, 37.2628, 35.645, 36.4722, 64.7507, 69.6705, 46.6985, 42.1247, 60.5934, 64.3711, 63.213, 67.8201, 37.6651, 35.9065, 36.8632, 35.5927, 66.4835, 57.9735, 51.8008, 47.8337, 49.2742, 50.8573, 45.6331, 47.6325, 37.4006, 44.0789, 36.6149, 42.791, 46.4838, 49.9537, 47.4119, 50.1232, 43.9127, 40.6913, 44.1479, 42.9418, 39.1408, 37.9875, 42.7109, 40.7974, 50.7508, 50.4153, 49.4463, 50.8464, 48.475, 64.0951, 46.355, 52.0532, 70.1456, 60.1016, 70.1456, 59.8417, 49.162, 50.8381, 49.0533, 50.764, 57.5077, 58.5324, 53.0601, 57.6885, 37.9481, 40.1218, 37.8652, 40.0436, 54.701, 52.3711, 52.3711, 55.9069, 77.3057, 57.4417, 41.6619, 38.7846, 38.7783]
    linear_psnr_rgb= [ 35.9487, 38.6972, 28.1428, 28.8158, 47.988, 38.0316, 31.2578, 29.4395, 51.1169, 44.6037, 45.3732, 38.8016,  38.2951, 36.0176, 33.9437, 35.1465, 58.5317, 56.1194, 45.4356, 40.9586, 52.5209, 53.536, 55.1065, 54.4791, 27.9428, 27.1324, 30.8219, 29.0177, 42.3658, 55.5694, 36.013, 37.3058, 36.3929, 28.386, 32.4035, 25.35, 27.1465, 37.598, 26.3466, 36.6442, 26.1536, 29.5237, 26.5428, 32.6665, 27.4188, 32.115, 26.2412, 32.9164, 31.0354, 28.5934, 38.3193, 39.0469, 36.2526, 41.6626, 35.886, 42.468, 29.9041, 42.806, 27.6514, 35.3595, 54.5473, 49.3884, 54.4829, 49.2934, 45.1081, 46.51, 43.8832, 45.9243, 52.8077, 51.4263, 48.0847, 50.8659, 26.8556, 29.7442, 27.4931, 32.2382, 54.701, 52.3711, 52.3711, 55.473, 77.3057, 56.457, 41.6421, 37.7842, 37.7792]
    log_psnr_rg= []
    pu_psnr_rgb= [ 55.549, 59.4349, 41.0087, 40.3667, 58.3946, 48.6654, 42.1934, 39.7274, 55.9724, 48.9066, 48.7227, 43.0063, 40.1223, 37.4897, 36.2513, 36.7548, 64.7507, 69.2243, 46.795, 42.2447, 60.5934, 64.3711, 63.213, 67.8201, 37.2742, 35.7568, 36.5175, 35.4817, 66.5136, 57.9803, 52.7034, 49.002, 48.7189, 50.4564, 45.0668, 46.5945, 36.9153, 44.2194, 36.1675, 42.9613, 45.8207, 49.5559, 46.7395, 49.8106, 44.4113, 41.5063, 44.7109, 43.8705, 39.0592, 38.2894, 43.0428, 41.5816, 50.8252, 50.7756, 49.8504, 51.2239, 48.7179, 63.72, 46.6267, 52.1256, 70.1472, 60.1567, 70.1472, 59.9068, 49.5263, 50.9401, 49.4077, 50.8721, 57.5127, 58.5419, 53.0669, 57.7047, 37.4826, 39.7814, 37.0909, 39.8143, 54.701, 52.3711, 52.3711, 57.0322, 77.3057, 58.0845, 41.7298, 39.1811, 39.1742]
    pq_psnr_rgb= [55.7007, 59.6376, 41.3956, 41.0335, 58.9619, 49.3322, 43.6997, 41.3974, 55.9697, 49.3607, 48.949, 44.1951, 40.1963, 37.5182, 36.3652, 36.7872, 64.7507, 69.9026, 46.8174, 42.3108, 60.5934, 64.3711, 63.213, 67.8201, 37.5753, 35.9552, 37.0525, 35.7948, 66.5136, 57.9722, 53.3624, 49.7142, 51.2487, 51.5093, 47.3105, 48.1585, 37.9534, 44.5369, 37.1103, 43.0422, 47.062, 49.9879, 47.8707, 50.307, 45.274, 41.9595, 45.9447, 44.8797, 39.2298, 38.5927, 43.2645, 41.9013, 52.3686, 50.9272, 51.3107, 51.3963, 50.8573, 65.2632, 48.4952, 54.0618, 70.1492, 60.2246, 70.1492, 59.9562, 49.8449, 51.1726, 49.7582, 51.1212, 57.532, 58.5614, 53.0808, 57.7334, 37.9238, 40.0783, 38.2703, 40.3644, 54.7169, 52.3851, 52.3851, 57.4529, 77.3057, 59.2326, 41.7608, 39.3312, 39.3239]
    navie_psnr_y= [58.1083, 61.9321, 42.5376, 42.9882, 67.2018, 55.0488, 44.8696, 42.7807, 78.581, 66.7458, 70.6341, 55.9083, 62.5465, 60.2105, 58.5885, 59.4198, 87.6252, 91.9225, 69.6423, 65.0276, 83.433, 87.2299, 86.113, 90.7299, 40.6235, 40.2304, 45.0869, 42.1724, 89.4199, 77.3432, 52.093, 48.4013, 53.5195, 52.7225, 50.6496, 48.9458, 51.2522, 61.329, 50.206, 65.4061, 49.3681, 53.2465, 50.5886, 55.4144, 47.641, 49.748, 46.3937, 49.1571, 42.3233, 42.0503, 52.1325, 53.8286, 52.6559, 60.9704, 50.9717, 61.3873, 51.6554, 66.8929, 48.3698, 53.7099, 93.0409, 82.1849, 93.0409, 82.554, 62.389, 62.7761, 61.116, 62.3625, 80.2765, 81.2461, 75.7444, 80.3115, 40.3691, 41.9213, 41.1204, 45.7828, 63.8688, 61.2053, 61.2053, 78.8546, 100.2534, 80.3894, 43.1869, 40.664, 40.6557]
    linear_psnr_y= [44.1063, 45.9772, 33.6813, 34.732, 52.8137, 43.732, 36.7003, 34.4923, 62.9951, 54.2309, 58.2635, 47.2279, 57.2737, 58.8989, 52.5496, 57.6663, 73.6707, 63.5345, 67.4011, 58.4476, 62.3889, 62.1328, 66.45, 59.0805, 31.5134, 30.5711, 36.0991, 33.7199, 51.12, 65.3197, 41.4262, 40.0003, 40.6237, 36.877, 36.4741, 30.815, 33.7804, 48.1707, 33.0506, 47.1844, 31.081, 34.6442, 30.7115, 40.0038, 29.4433, 39.4186, 28.1945, 36.8615, 34.253, 31.0128, 43.9334, 49.0622, 39.0982, 49.1504, 38.8289, 49.6722, 34.7483, 45.8838, 31.3294, 39.3632, 61.9886, 56.6904, 61.9681, 58.6725, 51.8877, 51.5583, 50.1467, 50.7087, 72.7239, 66.8197, 69.0236, 66.7339, 30.6258, 33.5345, 32.2007, 38.5948, 63.8688, 61.2053, 61.2053, 78.4207, 100.2534, 79.4047, 43.1504, 39.8111, 39.8043]
    log_psnr_y= []
    pu_psnr_y= [ 59.083, 63.5276, 42.9067, 43.2936, 68.6955, 56.7161, 46.4414, 44.0281, 78.8723, 67.2317, 70.5707, 56.8592, 63.0678, 60.4373, 59.194, 59.7025, 87.6252, 91.3369, 69.7389, 65.154, 83.433, 87.2299, 86.113, 90.7299, 40.0435, 39.7857, 44.7318, 41.9876, 89.4498, 77.5967, 53.268, 50.0089, 53.635, 52.5867, 50.6968, 48.4048, 50.8361, 61.9733, 49.7563, 65.5481, 49.1535, 52.8218, 50.1707, 54.9969, 48.0218, 50.7598, 47.0013, 50.37, 41.8859, 41.7572, 52.2915, 54.6248, 54.794, 62.4146, 53.043, 62.6835, 52.4313, 66.9629, 49.0876, 53.9925, 93.0439, 82.223, 93.0439, 82.6221, 62.678, 63.2431, 61.5534, 62.9426, 80.2824, 81.2564, 75.7526, 80.3316, 40.0244, 41.6144, 41.5753, 46.1767, 63.8688, 61.2053, 61.2053, 79.9799, 100.2534, 81.0322, 43.275, 40.9513, 40.9424]
    pq_psnr_y= [59.2114, 63.8829, 43.0525, 43.5334, 72.4724, 58.6662, 47.3227, 45.014, 78.8704, 68.0387, 71.0254, 58.6042, 63.1417, 60.4659, 59.3081, 59.7349, 87.6252, 92.2576, 69.7611, 65.224, 83.433, 87.2299, 86.113, 90.7299, 40.4147, 40.0846, 45.7982, 42.4299, 89.4498, 78.4583, 53.8727, 50.7905, 57.7715, 53.4206, 54.9667, 49.6062, 53.1728, 64.4232, 52.0209, 65.7011, 50.5409, 53.3141, 51.4023, 55.7851, 49.5137, 51.405, 48.7934, 52.9682, 42.1625, 42.1813, 53.044, 55.7184, 56.1211, 62.8523, 54.2466, 63.2942, 54.4868, 68.8053, 50.802, 56.0464, 93.0477, 82.3197, 93.0477, 82.6714, 65.9817, 66.3845, 64.4928, 65.9481, 80.3045, 81.279, 75.7704, 80.365, 40.3437, 41.8704, 42.0999, 46.6411, 63.9993, 61.3111, 61.3111, 80.4007, 100.2534, 82.1803, 43.3251, 41.1434, 41.1341]
    navie_cvvdp= [10.0, 10.0, 10.0, 10.0, 9.5466, 9.6603, 8.4055, 8.5068, 9.8201, 9.5003, 8.7256, 8.5287, 9.9532, 9.7599, 9.8921, 9.4489, 10.0, 10.0, 10.0, 10.0, 9.7688, 9.7561, 9.6914, 9.7317, 9.9906, 9.9958, 9.8885, 9.8351, 9.982, 9.988, 9.9864, 9.992, 8.2564, 8.1923, 8.8849, 8.4917, 9.9842, 9.8892, 9.2708, 9.0238, 9.3986, 9.4108, 9.2253, 9.1236, 9.3054, 9.6378, 9.2454, 9.841, 9.2664, 9.5717, 9.3413, 9.6011, 9.0068, 9.0318, 8.9101, 9.0588, 8.5039, 8.4181, 9.3603, 9.3227, 9.381, 9.6153, 9.2386, 9.6063, 9.3045, 9.7942, 9.0758, 9.3996, 9.9934, 9.9699, 9.9934, 9.9729, 9.6685, 9.6939, 9.6409, 9.6871, 9.9678, 9.9766, 9.9522, 9.9717, 8.2721, 8.5066, 8.5034, 8.8594, 9.7845, 9.7261, 9.7261, 9.9104, 9.9975, 9.9251, 8.7482, 8.4569, 8.4566]
    linear_cvvdp= [10.0, 10.0, 9.999, 9.9991, 8.5278, 8.7581, 6.6777, 7.068, 9.3541, 8.4869, 7.2308, 6.8021, 9.7726, 9.4587, 9.6442, 8.9721, 10.0, 10.0, 10.0, 10.0, 9.6051, 9.7352, 9.4206, 9.6737, 9.8867, 9.7609, 9.8506, 9.6433, 9.7563, 9.737, 9.8304, 9.6716, 7.7915, 7.7189, 7.818, 7.7857, 9.1722, 9.8085, 8.2118, 7.9158, 8.3544, 8.0643, 7.8453, 7.324, 7.91, 9.1293, 7.7351, 9.109, 7.8915, 8.3466, 8.1024, 8.8906, 7.5638, 8.3065, 7.036, 7.7646, 8.1192, 7.9865, 8.7598, 9.0131, 8.0055, 9.0369, 7.9345, 9.0571, 7.5622, 8.8547, 6.7367, 7.8786, 9.7121, 9.548, 9.7127, 9.5922, 9.2878, 9.3208, 9.1638, 9.2497, 9.9197, 9.8318, 9.9007, 9.8297, 7.5088, 7.8721, 6.7036, 7.9818, 9.7845, 9.7261, 9.7261, 9.8955, 9.9975, 9.8902, 8.7422, 8.3346, 8.335]
    log_cvvdp= []
    pu_cvvdp= [10.0, 10.0, 10.0, 10.0, 9.6159, 9.7315, 8.527, 8.6002, 9.851, 9.6214, 8.9961, 8.7828, 9.9567, 9.7946, 9.8932, 9.536, 10.0, 10.0, 10.0, 10.0, 9.8028, 9.7613, 9.7334, 9.745, 9.9906, 9.9951, 9.8971, 9.8489, 9.982, 9.988, 9.9864, 9.992, 7.9961, 7.9763, 8.7927, 8.4415, 9.9844, 9.909, 9.395, 9.2323, 9.4371, 9.4055, 9.2752, 9.062, 9.3064, 9.7645, 9.2355, 9.8535, 9.2284, 9.4796, 9.1526, 9.5672, 8.8288, 9.1702, 8.7798, 9.1907, 8.2968, 8.2528, 9.3561, 9.4471, 9.5423, 9.6994, 9.436, 9.6955, 9.3634, 9.7894, 9.1432, 9.4118, 9.9934, 9.9697, 9.9934, 9.9747, 9.725, 9.7297, 9.6969, 9.7223, 9.9679, 9.9769, 9.9524, 9.9719, 8.1115, 8.3751, 8.6491, 8.9659, 9.7845, 9.7261, 9.7261, 9.9346, 9.9975, 9.9296, 8.7633, 8.5106, 8.5101]
    pq_cvvdp= [10.0, 10.0, 10.0, 10.0, 9.6215, 9.7422, 8.553, 8.6406, 9.9186, 9.704, 9.0752, 8.8979, 9.9584, 9.8171, 9.9108, 9.6122, 10.0, 10.0, 10.0, 10.0, 9.8055, 9.7599, 9.7392, 9.7446, 9.9906, 9.9961, 9.9004, 9.853, 9.982, 9.988, 9.9864, 9.992, 8.1511, 8.102, 8.9525, 8.559, 9.985, 9.9139, 9.4323, 9.2905, 9.6054, 9.4881, 9.4893, 9.2104, 9.4568, 9.8089, 9.4141, 9.8605, 9.4198, 9.5681, 9.4463, 9.6481, 9.1898, 9.2356, 9.1489, 9.3776, 8.4057, 8.3904, 9.4482, 9.5473, 9.6164, 9.7268, 9.5059, 9.7285, 9.5154, 9.8341, 9.2987, 9.5387, 9.9934, 9.971, 9.9934, 9.9742, 9.7931, 9.799, 9.764, 9.7923, 9.9682, 9.9774, 9.9528, 9.9724, 8.2219, 8.4512, 8.7361, 9.0293, 9.7884, 9.7297, 9.7297, 9.9433, 9.9975, 9.9495, 8.7774, 8.559, 8.5584]

    # print(len(navie_psnr_rgb), len(linear_psnr_rgb), len(pu_psnr_rgb), len(pq_psnr_rgb  ))

    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(linear_cvvdp, pq_cvvdp)
    print('t statistic:', t_statistic)
    print('p value:', p_value)

    navie_psnr_rgb = np.array(navie_psnr_rgb)
    linear_psnr_rgb = np.array(linear_psnr_rgb)
    pu_psnr_rgb = np.array(pu_psnr_rgb)
    pq_psnr_rgb = np.array(pq_psnr_rgb)
    print(np.mean(navie_psnr_rgb), np.mean(linear_psnr_rgb), np.mean(pu_psnr_rgb), np.mean(pq_psnr_rgb))
    psnr_rgb_df = pd.DataFrame({'navie_psnr_rgb': navie_psnr_rgb, 'linear_psnr_rgb': linear_psnr_rgb, 'pu_psnr_rgb': pu_psnr_rgb, 'pq_psnr_rgb': pq_psnr_rgb})


    navie_psnr_y = np.array(navie_psnr_y)
    linear_psnr_y = np.array(linear_psnr_y)
    pu_psnr_y = np.array(pu_psnr_y)
    pq_psnr_y = np.array(pq_psnr_y)
    print(np.mean(navie_psnr_y), np.mean(linear_psnr_y), np.mean(pu_psnr_y), np.mean(pq_psnr_y))
    psnr_y_df = pd.DataFrame({ 'navie_psnr_y': navie_psnr_y , 'linear_psnr_y': linear_psnr_y, 'pu_psnr_y': pu_psnr_y, 'pq_psnr_y': pq_psnr_y})

    navie_cvvdp = np.array(navie_cvvdp)
    linear_cvvdp = np.array(linear_cvvdp)
    pu_cvvdp = np.array(pu_cvvdp)
    pq_cvvdp = np.array(pq_cvvdp)
    print(np.mean(navie_cvvdp), np.mean(linear_cvvdp), np.mean(pu_cvvdp), np.mean(pq_cvvdp))
    cvvdp_df = pd.DataFrame({'navie_cvvdp': navie_cvvdp, 'linear_cvvdp': linear_cvvdp, 'pu_cvvdp': pu_cvvdp, 'pq_cvvdp': pq_cvvdp})


    sns.violinplot(data=psnr_rgb_df)
    plt.ylabel('psnr_rgb')
    plt.savefig('psnr_rgb.png')
    plt.clf()

    sns.violinplot(data=psnr_y_df)
    plt.ylabel('psnr_y')
    plt.savefig('psnr_y.png')
    plt.clf()

    sns.violinplot(data=cvvdp_df)
    plt.ylabel('cvvdp')
    plt.savefig('cvvdp.png')
    plt.clf()

# ********************************************* WDSR *********************************************
if 1:
    navie_psnr_rgb= [55.2193, 58.5961, 57.3916, 41.0566, 40.673, 40.8837, 58.2001, 48.41, 50.1071, 42.2467, 40.0702, 39.8958, 55.6785, 48.9122, 48.8632, 48.7195, 43.3138, 43.3488, 39.6008, 37.2628, 36.2732, 35.645, 36.4722, 35.7564, 64.7507, 69.6705, 67.8539, 46.6985, 42.1247, 41.9698, 60.5934, 64.3711, 67.1057, 63.213, 67.8201, 68.8118, 37.6651, 35.9065, 35.8762, 36.8632, 35.5927, 35.7801, 66.4835, 57.9735, 58.0717, 51.8008, 47.8337, 47.9337, 49.2742, 50.8573, 49.9674, 45.6331, 47.6325, 47.1421, 37.4006, 44.0789, 45.859, 36.6149, 42.791, 44.2465, 46.4838, 49.9537, 48.9282, 47.4119, 50.1232, 49.0699, 43.9127, 40.6913, 41.4437, 44.1479, 42.9418, 43.3615, 39.1408, 37.9875, 38.3283, 42.7109, 40.7974, 41.646, 50.7508, 50.4153, 50.2499, 49.4463, 50.8464, 50.4924, 48.475, 64.0951, 67.9944, 46.355, 52.0532, 52.9892, 70.1456, 60.1016, 57.5846, 70.1456, 59.8417, 57.4371, 49.162, 50.8381, 51.4038, 49.0533, 50.764, 51.4618, 57.5077, 58.5324, 59.0522, 53.0601, 57.6885, 58.6058, 37.9481, 40.1218, 39.9317, 37.8652, 40.0436, 39.7021, 54.701, 52.3711, 54.701, 52.3711, 55.9069, 77.3057, 50.8075, 57.4417, 41.6619, 38.7846, 41.6084, 38.7783]
    linear_psnr_rgb= [31.9544, 33.0542, 32.821, 26.3938, 26.9375, 26.8726, 41.7933, 35.0459, 33.0152, 29.4978, 28.0154, 27.2787, 45.2114, 39.3589, 39.1088, 40.0409, 35.2988, 35.2286, 37.5495, 35.5329, 34.5153, 33.1053, 34.485, 33.7931, 44.3633, 42.7035, 42.6414, 40.4625, 37.8238, 37.7397, 41.6623, 43.3522, 43.8984, 42.1902, 43.1981, 43.6351, 27.017, 26.3974, 25.4821, 29.3682, 28.1282, 26.9386, 38.1744, 48.9657, 49.0633, 31.5244, 33.617, 33.7534, 33.4069, 27.3252, 26.1936, 30.4992, 24.7749, 24.0675, 26.4116, 33.9079, 34.9042, 25.7523, 33.4585, 34.4899, 25.503, 28.3925, 26.718, 25.9332, 30.9116, 28.7101, 26.5015, 30.8031, 29.6657, 25.3388, 30.6494, 29.8363, 28.6573, 27.601, 26.897, 34.4008, 35.3232, 34.8264, 33.0857, 35.075, 34.3326, 32.5835, 36.0449, 35.051, 27.9516, 37.5318, 38.226, 26.0683, 31.147, 31.3301, 42.3793, 39.7442, 39.7504, 41.9961, 39.9426, 40.0352, 42.1047, 42.0906, 41.9959, 40.6809, 41.4038, 41.8618, 44.782, 41.8733, 41.2485, 41.0758, 40.62, 40.3092, 25.7793, 27.706, 27.4181, 26.6023, 29.17, 28.7447, 54.701, 52.3676, 54.701, 52.3676, 55.3675, 77.3057, 50.1939, 56.4332, 41.6184, 37.717, 41.5655, 37.712]
    log_psnr_rg= []
    pu_psnr_rgb= [51.8035, 55.9151, 54.5905, 37.7929, 37.0112, 37.1635, 54.3249, 44.0533, 42.9627, 36.3993, 34.4942, 34.0097, 54.9508, 46.62, 46.5412, 47.4829, 40.5826, 40.5575, 39.6274, 37.3344, 36.3654, 35.6949, 36.5393, 35.844, 64.7507, 62.3563, 61.9356, 46.6888, 42.0165, 41.8574, 60.5934, 64.3711, 67.1003, 63.213, 67.8201, 68.8118, 35.0385, 33.833, 33.5442, 33.8034, 33.3512, 33.0874, 66.5135, 57.8422, 57.9373, 48.294, 44.0702, 44.1468, 42.7444, 42.4055, 41.0087, 39.3272, 36.956, 36.2437, 34.0549, 43.1502, 44.8816, 33.0695, 42.0114, 43.605, 40.2534, 44.8406, 42.4876, 41.1483, 45.3119, 43.5352, 40.6006, 40.2996, 40.8645, 40.0199, 41.1758, 41.4135, 37.6203, 36.7531, 36.8946, 40.7617, 40.6335, 41.3414, 43.4487, 48.7727, 47.6804, 42.8439, 49.0762, 47.5299, 41.6373, 55.4714, 56.8799, 40.0088, 44.6971, 45.2821, 70.1046, 59.2773, 57.0533, 70.1046, 59.4886, 57.1792, 47.7688, 49.2908, 49.96, 47.3543, 49.0758, 50.185, 57.5044, 58.449, 58.9582, 52.9237, 57.4053, 58.3492, 34.7834, 37.7145, 37.527, 32.8747, 36.6425, 36.3184, 54.701, 52.3665, 54.701, 52.3688, 56.325, 75.793, 51.2299, 57.6311, 41.6913, 38.9799, 41.6375, 38.9733]
    pq_psnr_rgb= [54.1844, 58.6155, 57.2595, 40.6504, 40.3854, 40.5667, 58.3779, 48.5755, 50.1856, 42.549, 40.2718, 40.0879, 55.704, 49.0063, 48.9538, 48.7721, 43.7769, 43.8165, 40.067, 37.4686, 36.526, 36.2166, 36.7244, 36.0318, 64.7507, 69.0938, 67.4658, 46.797, 42.2608, 42.0979, 60.5934, 64.3711, 67.1057, 63.213, 67.8201, 68.8118, 37.0275, 35.5013, 35.4632, 36.437, 35.3407, 35.4946, 66.5136, 57.9756, 58.0738, 52.3906, 48.6518, 48.7427, 49.9331, 48.8298, 47.6504, 46.3254, 45.5093, 44.9353, 37.4944, 44.2907, 46.1644, 36.5693, 42.7962, 44.3054, 45.6276, 48.5935, 47.5302, 46.5495, 49.1677, 47.9273, 44.3447, 41.5153, 42.2208, 44.7852, 44.2711, 44.6181, 38.7395, 37.9853, 38.2884, 42.715, 41.2784, 42.1672, 50.2251, 50.5158, 50.3052, 49.3682, 50.9444, 50.535, 48.1774, 62.6098, 65.1468, 46.3952, 51.4507, 52.2176, 70.1426, 60.1223, 57.6328, 70.1426, 59.9254, 57.5207, 49.4111, 50.9319, 51.5258, 49.2962, 50.8673, 51.5825, 57.5134, 58.551, 59.0731, 53.0667, 57.7097, 58.6292, 37.3734, 39.5121, 39.3231, 37.6198, 39.728, 39.3291, 54.701, 52.3711, 54.701, 52.3711, 57.0492, 77.3057, 52.0026, 58.5454, 41.7259, 39.1875, 41.6717, 39.1806]
    navie_psnr_y= [58.1083, 61.9321, 60.3992, 42.5376, 42.9882, 43.2155, 67.2018, 55.0488, 53.0717, 44.8696, 42.7807, 41.8767, 78.581, 66.7458, 66.7198, 70.6341, 55.9083, 55.7115, 62.5465, 60.2105, 59.2209, 58.5885, 59.4198, 58.7041, 87.6252, 91.9225, 90.328, 69.6423, 65.0276, 64.8729, 83.433, 87.2299, 90.0005, 86.113, 90.7299, 91.7112, 40.6235, 40.2304, 39.3996, 45.0869, 42.1724, 41.1308, 89.4199, 77.3432, 77.386, 52.093, 48.4013, 48.5036, 53.5195, 52.7225, 51.6363, 50.6496, 48.9458, 48.3692, 51.2522, 61.329, 61.7424, 50.206, 65.4061, 66.8312, 49.3681, 53.2465, 51.2298, 50.5886, 55.4144, 52.8455, 47.641, 49.748, 50.1151, 46.3937, 49.1571, 49.2474, 42.3233, 42.0503, 41.8067, 52.1325, 53.8286, 52.8821, 52.6559, 60.9704, 58.9078, 50.9717, 61.3873, 59.0287, 51.6554, 66.8929, 74.2876, 48.3698, 53.7099, 54.9139, 93.0409, 82.1849, 79.9651, 93.0409, 82.554, 80.1828, 62.389, 62.7761, 66.291, 61.116, 62.3625, 66.3857, 80.2765, 81.2461, 81.7567, 75.7444, 80.3115, 81.2555, 40.3691, 41.9213, 41.7047, 41.1204, 45.7828, 45.1482, 63.8688, 61.2053, 63.8688, 61.2053, 78.8546, 100.2534, 73.7552, 80.3894, 43.1869, 40.664, 43.1334, 40.6557]
    linear_psnr_y= [32.7206, 33.7599, 33.5168, 27.502, 28.1326, 28.0502, 43.312, 36.48, 34.2303, 30.9591, 29.5385, 28.7106, 50.959, 42.4132, 42.2457, 44.5312, 38.1423, 38.0802, 45.7616, 52.8049, 51.9651, 41.2066, 49.4782, 49.2602, 54.2489, 47.9932, 47.9166, 53.2355, 44.9384, 44.8457, 48.5776, 47.7374, 48.0825, 49.3756, 48.328, 48.7166, 28.3823, 27.8426, 26.6713, 31.6387, 30.4118, 28.7339, 39.854, 51.3447, 51.423, 32.4695, 34.2803, 34.4023, 35.6466, 29.6953, 28.7727, 32.2436, 26.7631, 26.213, 30.9507, 40.8315, 41.4818, 30.0833, 39.1268, 39.7374, 27.2034, 30.0547, 28.2484, 27.4571, 32.8626, 30.4577, 27.0336, 34.9132, 32.2338, 25.762, 32.3791, 31.3062, 29.6553, 28.5304, 27.6674, 37.1615, 39.605, 38.1414, 33.948, 36.3381, 35.4986, 33.5039, 37.4096, 36.2952, 28.95, 38.3087, 39.131, 26.9351, 31.9039, 32.0797, 45.2207, 42.1929, 42.1089, 44.7055, 42.52, 42.5654, 46.0488, 45.9461, 45.5278, 44.1694, 45.3195, 45.6707, 53.3681, 51.84, 49.0334, 50.0491, 50.669, 48.6902, 26.9606, 28.616, 28.2576, 28.3754, 30.3708, 29.9714, 63.8688, 61.2043, 63.8688, 61.2043, 78.3152, 100.2534, 73.1417, 79.3809, 43.1456, 39.788, 43.0927, 39.7812]
    log_psnr_y= []
    pu_psnr_y= [56.5143, 59.7701, 58.3892, 41.3298, 41.6297, 41.7831, 59.8305, 50.0238, 47.9067, 42.055, 39.7538, 38.8515, 77.5402, 59.9138, 59.8891, 65.5317, 51.5959, 51.3327, 62.5723, 60.2821, 59.313, 58.6262, 59.4869, 58.7917, 87.6252, 80.2186, 79.325, 69.6327, 64.5433, 64.3974, 83.433, 87.2299, 89.9921, 86.113, 90.7299, 91.7112, 38.5221, 38.2, 37.3029, 41.0813, 39.8668, 38.5351, 89.4468, 74.2012, 74.2139, 50.2147, 46.2076, 46.2829, 47.7831, 46.7802, 45.6042, 44.5854, 42.3542, 41.6841, 44.0731, 58.0899, 58.3323, 42.7404, 61.6877, 63.6149, 44.8667, 49.3501, 46.6291, 46.1023, 50.6736, 48.5199, 43.867, 48.7499, 48.7882, 42.801, 45.8584, 45.8946, 40.59, 40.3534, 40.0909, 47.9439, 52.6338, 51.6219, 47.393, 56.9376, 54.4504, 46.6021, 57.5312, 54.4168, 46.058, 60.168, 62.0344, 43.8135, 47.8621, 48.6021, 92.2784, 76.1247, 75.4361, 92.2784, 81.9545, 79.7601, 57.9984, 58.3646, 60.5177, 56.2856, 57.7722, 61.4362, 80.272, 81.1419, 81.6395, 75.5715, 79.9543, 80.9319, 38.4478, 40.2662, 40.0472, 38.3609, 43.3783, 42.7651, 63.8688, 61.1944, 63.8688, 61.2043, 78.7503, 84.5385, 74.1777, 80.4984, 43.2407, 40.7366, 43.1866, 40.7283]
    pq_psnr_y= [57.5251, 62.4036, 60.7177, 42.5332, 43.0487, 43.235, 69.5304, 56.6877, 54.0005, 46.4886, 44.0938, 43.0988, 78.6065, 66.6703, 66.6425, 70.393, 57.1787, 57.0209, 63.0122, 60.4163, 59.4737, 59.1588, 59.6721, 58.9794, 87.6252, 91.0209, 89.6841, 69.7408, 65.1546, 64.9922, 83.433, 87.2299, 90.0005, 86.113, 90.7299, 91.7112, 40.0908, 39.8438, 39.0258, 45.0122, 42.0968, 40.9617, 89.4498, 78.3267, 78.3804, 53.3425, 50.0854, 50.1802, 56.3684, 49.7337, 48.3158, 53.4277, 46.8546, 46.1054, 50.9557, 63.5366, 64.2427, 49.5119, 65.0761, 66.6211, 48.2938, 51.0789, 49.503, 49.328, 53.2049, 51.0531, 48.387, 50.8118, 50.8027, 47.5199, 51.6197, 51.3712, 41.7976, 41.6576, 41.4313, 51.7435, 53.7559, 52.903, 54.3356, 61.6483, 59.6176, 52.8748, 62.0316, 59.7426, 51.0888, 66.0477, 70.1137, 48.6627, 53.6759, 54.6908, 93.0356, 80.9214, 79.1912, 93.0356, 82.6344, 80.2638, 64.2732, 64.9638, 67.5164, 62.7773, 64.5227, 68.2173, 80.2831, 81.2664, 81.7795, 75.7514, 80.3332, 81.2795, 39.9749, 41.5025, 41.2889, 41.629, 46.1328, 45.3286, 63.8688, 61.2053, 63.8688, 61.2053, 79.9969, 100.2534, 74.9504, 81.4932, 43.2832, 40.939, 43.2286, 40.9301]
    navie_cvvdp= [9.5466, 9.6603, 9.6156, 8.4055, 8.5068, 8.5345, 9.8201, 9.5003, 9.4234, 8.7256, 8.5287, 8.4476, 9.9532, 9.7599, 9.7599, 9.8921, 9.4489, 9.4426, 9.7688, 9.7561, 9.7308, 9.6914, 9.7317, 9.7136, 9.9906, 9.9958, 9.9943, 9.8885, 9.8351, 9.8357, 9.982, 9.988, 9.9921, 9.9864, 9.992, 9.993, 8.2564, 8.1923, 8.0384, 8.8849, 8.4917, 8.3384, 9.9842, 9.8892, 9.8895, 9.2708, 9.0238, 9.0348, 9.3986, 9.4108, 9.363, 9.2253, 9.1236, 9.085, 9.3054, 9.6378, 9.6381, 9.2454, 9.841, 9.859, 9.2664, 9.5717, 9.4567, 9.3413, 9.6011, 9.493, 9.0068, 9.0318, 9.0729, 8.9101, 9.0588, 9.0722, 8.5039, 8.4181, 8.3761, 9.3603, 9.3227, 9.2966, 9.381, 9.6153, 9.5731, 9.2386, 9.6063, 9.5641, 9.3045, 9.7942, 9.8929, 9.0758, 9.3996, 9.4449, 9.9934, 9.9699, 9.9539, 9.9934, 9.9729, 9.9552, 9.6685, 9.6939, 9.7702, 9.6409, 9.6871, 9.7741, 9.9678, 9.9766, 9.978, 9.9522, 9.9717, 9.975, 8.2721, 8.5066, 8.4689, 8.5034, 8.8594, 8.826, 9.7845, 9.7261, 9.7845, 9.7261, 9.9104, 9.9975, 9.8725, 9.9251, 8.7482, 8.4569, 8.7436, 8.4566]
    linear_cvvdp= [8.4003, 8.5988, 8.5213, 6.4911, 6.8778, 6.8226, 9.1401, 8.3087, 7.9075, 7.0751, 6.6003, 6.3154, 9.4828, 8.9821, 8.9672, 9.1221, 8.5217, 8.5086, 9.177, 9.5727, 9.5438, 8.8041, 9.4231, 9.4165, 9.6058, 9.3517, 9.3478, 9.5752, 9.2062, 9.2, 9.3732, 9.3326, 9.3524, 9.4097, 9.3591, 9.3792, 7.4537, 7.4399, 7.2453, 7.3709, 7.439, 7.1751, 8.9557, 9.4968, 9.5083, 8.0357, 7.7828, 7.7934, 8.1587, 7.6661, 7.5567, 7.6034, 6.8237, 6.7438, 7.6311, 8.7929, 8.8532, 7.418, 8.637, 8.6995, 7.5733, 8.0586, 7.7741, 7.8006, 8.5523, 8.3625, 7.4037, 8.2341, 8.048, 6.836, 7.6052, 7.5033, 7.7617, 7.7175, 7.6621, 8.367, 8.6437, 8.5657, 7.8248, 8.528, 8.4051, 7.7488, 8.6298, 8.4801, 7.2293, 8.6535, 8.7566, 6.3885, 7.6672, 7.8775, 9.2175, 9.0399, 9.042, 9.1784, 9.0643, 9.0769, 9.1583, 9.2337, 9.2149, 9.0204, 9.1677, 9.2251, 9.5701, 9.5228, 9.3978, 9.4373, 9.4709, 9.379, 7.239, 7.5711, 7.5168, 6.3886, 7.6724, 7.5778, 9.7845, 9.7261, 9.7845, 9.7261, 9.8921, 9.9975, 9.8388, 9.8879, 8.7435, 8.3279, 8.7389, 8.3284]
    log_cvvdp= []
    pu_cvvdp= [9.5596, 9.6857, 9.6412, 8.3948, 8.4841, 8.507, 9.6986, 9.3782, 9.2551, 8.6871, 8.4205, 8.3134, 9.9579, 9.6569, 9.6569, 9.8265, 9.3539, 9.3406, 9.7957, 9.7954, 9.7793, 9.7331, 9.7715, 9.7615, 9.9906, 9.9609, 9.9589, 9.8902, 9.8359, 9.8353, 9.982, 9.988, 9.9921, 9.9864, 9.992, 9.993, 8.0249, 7.9955, 7.8311, 8.5579, 8.34, 8.1386, 9.9844, 9.8894, 9.8896, 9.2982, 9.0528, 9.06, 9.1965, 9.206, 9.1072, 8.9279, 8.7658, 8.677, 8.8748, 9.6573, 9.659, 8.7695, 9.7731, 9.8165, 8.9557, 9.37, 9.0696, 8.9878, 9.4408, 9.2235, 8.5487, 9.1864, 9.224, 8.4975, 9.0848, 9.0981, 8.2761, 8.2494, 8.1908, 9.2288, 9.3578, 9.2948, 9.2099, 9.5997, 9.5392, 9.1398, 9.5949, 9.5134, 9.148, 9.7018, 9.732, 8.9667, 9.2352, 9.2734, 9.9933, 9.9434, 9.9367, 9.9933, 9.9729, 9.9554, 9.597, 9.6492, 9.6881, 9.5569, 9.6407, 9.7096, 9.9681, 9.9768, 9.9782, 9.952, 9.9717, 9.9752, 8.0632, 8.3307, 8.2877, 8.3092, 8.8388, 8.7831, 9.7845, 9.726, 9.7845, 9.7261, 9.907, 9.9478, 9.8901, 9.9278, 8.7586, 8.4887, 8.7539, 8.4882]
    pq_cvvdp= [9.6192, 9.7351, 9.6922, 8.5484, 8.6127, 8.639, 9.8909, 9.6773, 9.5866, 9.0637, 8.8588, 8.783, 9.9601, 9.8024, 9.8023, 9.8974, 9.5896, 9.5858, 9.813, 9.7828, 9.7662, 9.7542, 9.7657, 9.7535, 9.9906, 9.9956, 9.9941, 9.899, 9.8542, 9.854, 9.982, 9.988, 9.9921, 9.9864, 9.992, 9.993, 8.125, 8.0845, 7.9199, 8.9111, 8.5511, 8.3471, 9.9844, 9.9202, 9.9206, 9.4265, 9.2738, 9.2809, 9.5999, 9.4495, 9.385, 9.4851, 9.1586, 9.1146, 9.4318, 9.8123, 9.818, 9.3789, 9.8623, 9.8773, 9.4019, 9.5514, 9.4303, 9.4315, 9.627, 9.483, 9.1741, 9.2917, 9.335, 9.1259, 9.393, 9.4061, 8.3798, 8.3653, 8.3164, 9.4067, 9.5174, 9.4433, 9.5886, 9.7259, 9.7037, 9.4901, 9.7176, 9.6921, 9.4797, 9.8191, 9.8745, 9.2925, 9.5134, 9.5417, 9.9934, 9.96, 9.9513, 9.9934, 9.9745, 9.9586, 9.7722, 9.7857, 9.8343, 9.7452, 9.7794, 9.8457, 9.9679, 9.9772, 9.9787, 9.9524, 9.9723, 9.9757, 8.1831, 8.4142, 8.3752, 8.713, 9.0129, 8.9499, 9.7845, 9.7261, 9.7845, 9.7261, 9.9377, 9.9975, 9.91, 9.9422, 8.7663, 8.527, 8.7615, 8.5264]

    # print(len(navie_psnr_rgb), len(linear_psnr_rgb), len(pu_psnr_rgb), len(pq_psnr_rgb  ))

    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(linear_cvvdp, pq_cvvdp)
    print('t statistic:', t_statistic)
    print('p value:', p_value)

    navie_psnr_rgb = np.array(navie_psnr_rgb)
    linear_psnr_rgb = np.array(linear_psnr_rgb)
    pu_psnr_rgb = np.array(pu_psnr_rgb)
    pq_psnr_rgb = np.array(pq_psnr_rgb)
    print(np.mean(navie_psnr_rgb), np.mean(linear_psnr_rgb), np.mean(pu_psnr_rgb), np.mean(pq_psnr_rgb))
    psnr_rgb_df = pd.DataFrame({'navie_psnr_rgb': navie_psnr_rgb, 'linear_psnr_rgb': linear_psnr_rgb, 'pu_psnr_rgb': pu_psnr_rgb, 'pq_psnr_rgb': pq_psnr_rgb})


    navie_psnr_y = np.array(navie_psnr_y)
    linear_psnr_y = np.array(linear_psnr_y)
    pu_psnr_y = np.array(pu_psnr_y)
    pq_psnr_y = np.array(pq_psnr_y)
    print(np.mean(navie_psnr_y), np.mean(linear_psnr_y), np.mean(pu_psnr_y), np.mean(pq_psnr_y))
    psnr_y_df = pd.DataFrame({ 'navie_psnr_y': navie_psnr_y , 'linear_psnr_y': linear_psnr_y, 'pu_psnr_y': pu_psnr_y, 'pq_psnr_y': pq_psnr_y})

    navie_cvvdp = np.array(navie_cvvdp)
    linear_cvvdp = np.array(linear_cvvdp)
    pu_cvvdp = np.array(pu_cvvdp)
    pq_cvvdp = np.array(pq_cvvdp)
    print(np.mean(navie_cvvdp), np.mean(linear_cvvdp), np.mean(pu_cvvdp), np.mean(pq_cvvdp))
    cvvdp_df = pd.DataFrame({'navie_cvvdp': navie_cvvdp, 'linear_cvvdp': linear_cvvdp, 'pu_cvvdp': pu_cvvdp, 'pq_cvvdp': pq_cvvdp})


    sns.violinplot(data=psnr_rgb_df)
    plt.ylabel('wdsr_psnr_rgb')
    plt.savefig('wdsr_psnr_rgb.png')
    plt.clf()

    sns.violinplot(data=psnr_y_df)
    plt.ylabel('wdsr_psnr_y')
    plt.savefig('wdsr_psnr_y.png')
    plt.clf()

    sns.violinplot(data=cvvdp_df)
    plt.ylabel('wdsr_cvvdp')
    plt.savefig('wdsr_cvvdp.png')
    plt.clf()



# ********************************************* DnCNN *********************************************
if 1:
    navie_psnr_rgb= []
    linear_psnr_rgb= [56.7926, 56.4013, 49.2621, 50.1064, 49.9638, 57.595, 47.4354, 43.7179, 42.7617, 39.2092, 37.8296, 50.6808, 50.4203, 50.4825, 47.8852, 45.4659, 45.5624, 38.5403, 36.2366, 35.1004, 34.1518, 35.3828, 34.5499, 64.3443, 69.9684, 67.8037, 46.275, 41.7589, 41.5827, 60.2092, 63.9755, 66.6362, 62.644, 67.4529, 68.4127, 32.1689, 29.8873, 28.662, 34.0228, 31.2137, 29.9821, 65.7968, 58.0029, 58.1017, 57.1133, 53.9517, 54.0413, 36.7791, 29.1385, 27.7339, 32.6005, 25.5502, 24.7357, 27.6249, 36.7339, 38.6908, 26.8313, 36.3113, 38.2023, 27.6446, 31.7192, 28.9396, 27.5972, 36.1688, 31.8302, 27.1912, 32.1847, 31.6581, 26.101, 33.4567, 33.2209, 39.6176, 32.8487, 31.7859, 40.9354, 38.6142, 39.3575, 46.9971, 46.8727, 46.9833, 46.9126, 47.125, 47.1479, 38.3292, 62.831, 71.4324, 34.63, 50.9564, 53.6396, 68.6527, 58.7106, 56.2276, 68.6527, 56.5705, 54.8937, 46.7545, 47.1338, 47.3735, 46.718, 47.1968, 47.4462, 56.178, 56.9593, 57.4245, 51.9282, 56.4057, 57.1481, 29.3358, 33.6854, 32.6866, 29.5568, 38.2223, 37.4455, 54.701, 52.3711, 54.701, 52.3711, 55.3954, 77.3057, 50.2371, 56.4243, 41.6424, 37.7375, 41.5891, 37.7325]
    log_psnr_rg= []
    pu_psnr_rgb= [63.5036, 63.5036, 76.2447, 63.3081, 63.3084, 57.9488, 47.6288, 43.8296, 42.8768, 39.3029, 37.9088, 50.7288, 50.6484, 50.6274, 47.9325, 45.6039, 45.6759, 38.5457, 36.2778, 35.1366, 34.1147, 35.4295, 34.595, 64.3443, 69.9684, 67.8037, 46.2562, 41.7309, 41.5585, 60.2732, 64.0262, 66.6362, 62.7189, 67.5667, 68.5071, 32.7366, 30.697, 29.5055, 34.1855, 31.957, 30.7942, 66.4838, 58.003, 58.1019, 63.002, 57.089, 57.1691, 38.1613, 33.3996, 32.0798, 34.1117, 28.7487, 28.0276, 28.387, 37.5418, 39.0951, 27.7943, 37.0917, 38.7098, 29.2817, 33.5495, 30.6125, 29.1317, 38.4295, 33.7308, 28.115, 32.1866, 31.8938, 27.152, 33.4044, 33.1947, 39.739, 33.1672, 32.1904, 41.7652, 38.7239, 39.5518, 47.069, 49.8885, 50.085, 46.9877, 50.2454, 50.3098, 38.6997, 61.9759, 69.2385, 35.0187, 50.0635, 52.2231, 68.8033, 58.649, 56.1955, 68.8033, 58.4468, 56.0794, 46.1408, 48.0783, 48.8047, 46.1343, 48.0728, 48.8036, 56.1864, 57.003, 57.4731, 51.9356, 56.4411, 57.1902, 31.0648, 34.3485, 33.5694, 31.6463, 37.7854, 37.2254, 54.701, 52.3711, 54.701, 52.3711, 55.3954, 77.3057, 50.2628, 56.4243, 41.6404, 37.6951, 41.5872, 37.6901]
    pq_psnr_rgb= [62.0213, 61.9038, 57.2251, 57.4705, 57.1615, 58.0207, 47.5876, 43.994, 42.8998, 39.3682, 38.0183, 54.4973, 49.7918, 49.7809, 49.509, 45.3157, 45.3853, 39.5615, 37.2197, 36.2971, 35.7821, 36.4658, 35.7986, 64.3443, 69.9684, 67.8037, 46.393, 41.8453, 41.6562, 60.2732, 64.0262, 66.6362, 62.7189, 67.5667, 68.5071, 32.9254, 31.0164, 29.9557, 34.1025, 32.1267, 31.1324, 66.4838, 58.003, 58.1019, 62.2574, 56.7996, 56.8745, 39.4615, 36.1903, 34.9897, 35.6526, 31.4887, 30.8338, 30.336, 42.4432, 45.2149, 29.4765, 41.2127, 43.8912, 30.3732, 34.7147, 31.8245, 30.2727, 39.3518, 34.9845, 28.8981, 32.2521, 31.9557, 28.0456, 33.362, 33.04, 38.3202, 33.3891, 32.5011, 40.9386, 39.0229, 39.7421, 46.9243, 49.716, 49.8489, 46.8391, 50.0739, 50.0705, 39.015, 60.4582, 67.023, 35.4764, 47.7338, 49.0542, 68.8033, 58.5883, 56.1609, 68.8033, 58.4404, 56.0757, 43.7094, 43.3085, 45.1978, 43.699, 43.3061, 45.1973, 56.1864, 57.003, 57.4731, 51.9356, 56.4411, 57.1902, 32.2689, 35.1595, 34.5369, 32.5515, 39.2734, 38.6058, 54.701, 52.3711, 54.701, 52.3711, 56.1496, 77.3057, 50.9133, 57.0526, 41.6801, 38.3327, 41.6264, 38.327]
    navie_psnr_y= []
    linear_psnr_y= [77.8626, 77.3059, 69.402, 70.063, 69.9739, 63.4956, 54.6612, 49.1456, 49.6425, 44.5274, 42.8466, 73.6284, 73.2622, 73.3219, 70.8069, 64.9598, 64.7396, 61.488, 59.1843, 58.0481, 57.0955, 58.3305, 57.4976, 87.2887, 92.9161, 90.7514, 69.2227, 64.7063, 64.5302, 83.1531, 86.9197, 89.584, 85.5914, 90.4007, 91.3604, 34.6006, 32.5832, 30.6915, 38.8726, 35.5188, 33.0646, 87.669, 80.9505, 81.0493, 65.0584, 63.8305, 63.8361, 39.4072, 35.7053, 34.6738, 34.8627, 28.8025, 28.2286, 34.3284, 59.3257, 61.153, 33.9597, 59.1881, 61.1184, 30.8825, 34.9694, 31.6103, 30.1265, 42.0664, 35.4742, 27.6991, 39.5245, 36.6697, 26.5567, 36.4156, 36.1485, 45.0917, 35.1371, 33.4643, 53.6329, 52.3955, 52.6887, 48.3651, 64.2336, 64.3202, 48.2671, 64.3019, 64.1684, 43.0976, 70.6352, 89.2171, 38.1552, 55.1849, 58.3787, 91.6004, 81.6515, 79.1702, 91.6004, 79.5141, 77.8377, 69.7015, 69.684, 69.9032, 69.6627, 69.7385, 69.9691, 79.1245, 79.9058, 80.3709, 74.8666, 79.3435, 80.0911, 31.5458, 35.8315, 34.3921, 32.8231, 42.5632, 42.2317, 63.8688, 61.2053, 63.8688, 61.2053, 78.3431, 100.2534, 73.1848, 79.3721, 43.149, 39.7995, 43.096, 39.7927]
    log_psnr_y= []
    pu_psnr_y= [86.4513, 86.4513, 97.8182, 86.1755, 86.1761, 63.5324, 54.6723, 49.1586, 49.6567, 44.5369, 42.8561, 73.6765, 73.5199, 73.4987, 70.8572, 65.0275, 64.7924, 61.4934, 59.2255, 58.0843, 57.0618, 58.3772, 57.5428, 87.2887, 92.9161, 90.7514, 69.2039, 64.6773, 64.5049, 83.2171, 86.9704, 89.584, 85.6663, 90.5144, 91.4548, 34.7446, 32.7127, 30.8188, 38.9456, 35.6098, 33.1738, 89.4315, 80.9507, 81.0496, 65.2214, 63.9894, 63.9922, 39.6946, 36.2602, 35.2731, 35.1419, 29.1955, 28.6369, 34.4703, 60.0695, 61.52, 34.1118, 59.9568, 61.6224, 31.1117, 35.2362, 31.8563, 30.339, 42.48, 35.7747, 27.8898, 39.6596, 36.9128, 26.7576, 36.5987, 36.3617, 45.2168, 35.23, 33.5578, 53.6631, 52.3011, 52.5901, 48.3807, 64.8795, 64.9746, 48.2825, 64.9354, 64.7867, 43.1902, 70.6091, 87.8876, 38.235, 55.2466, 58.4549, 91.751, 81.5452, 79.1126, 91.751, 81.3865, 79.0213, 69.0881, 70.5378, 71.1822, 69.0803, 70.53, 71.1812, 79.1329, 79.9495, 80.4195, 74.8739, 79.3788, 80.1331, 31.8676, 36.2004, 34.7328, 33.1031, 42.8749, 42.5621, 63.8688, 61.2053, 63.8688, 61.2053, 78.3431, 100.2534, 73.2105, 79.3721, 43.1463, 39.7649, 43.0934, 39.7582]
    pq_psnr_y= [84.6267, 84.4797, 78.357, 79.0821, 78.7552, 63.6459, 54.6992, 49.1794, 49.6727, 44.563, 42.8828, 77.445, 72.6764, 72.6653, 72.4236, 64.8969, 64.6697, 62.5093, 60.1674, 59.2448, 58.7289, 59.4135, 58.7463, 87.2887, 92.9161, 90.7514, 69.3407, 64.7919, 64.6029, 83.2171, 86.9704, 89.584, 85.6663, 90.5144, 91.4548, 35.0124, 32.9969, 31.1168, 39.1842, 35.8657, 33.4444, 89.4315, 80.9507, 81.0496, 65.2891, 63.9861, 63.9888, 40.972, 38.4721, 37.5208, 36.7076, 31.5555, 31.0247, 34.8411, 64.2064, 66.3314, 34.4781, 63.9526, 66.7195, 31.7435, 35.9213, 32.5519, 31.0136, 42.9755, 36.4988, 28.8488, 40.0944, 37.6849, 27.7662, 37.108, 36.8858, 45.1634, 35.4615, 33.8025, 53.6053, 52.3599, 52.5519, 48.4339, 64.8455, 64.9264, 48.3344, 64.9057, 64.7519, 43.3263, 70.4151, 86.2478, 38.4003, 54.773, 57.611, 91.751, 81.4429, 79.0539, 91.751, 81.3754, 79.0148, 66.6546, 66.0873, 67.8879, 66.6398, 66.0834, 67.8874, 79.1329, 79.9495, 80.4195, 74.8739, 79.3788, 80.1331, 33.3634, 37.5594, 36.1805, 34.2744, 44.029, 43.5877, 63.8688, 61.2053, 63.8688, 61.2053, 79.0973, 100.2534, 73.861, 80.0004, 43.2119, 40.7026, 43.1581, 40.6942]
    navie_cvvdp= []
    linear_cvvdp= [9.9546, 9.9539, 9.9135, 9.9091, 9.9091, 9.6385, 9.2305, 8.8476, 8.8382, 8.3218, 8.0788, 9.8165, 9.8961, 9.8968, 9.8163, 9.7243, 9.7115, 9.6837, 9.7378, 9.7176, 9.5317, 9.6878, 9.6791, 9.99, 9.9959, 9.9939, 9.8816, 9.819, 9.8173, 9.9813, 9.9876, 9.9917, 9.9857, 9.992, 9.9931, 7.9751, 7.7424, 7.4678, 7.9272, 7.8826, 7.5806, 9.9811, 9.96, 9.9611, 9.6273, 9.6097, 9.6095, 8.157, 7.6969, 7.6054, 7.5431, 6.7394, 6.6538, 7.9579, 9.599, 9.6417, 7.8448, 9.6101, 9.6685, 7.7745, 8.2551, 7.9432, 7.9718, 8.9213, 8.7195, 7.4898, 8.3208, 8.2156, 6.9328, 7.675, 7.6228, 8.9216, 8.1439, 8.0177, 9.3262, 9.067, 9.0734, 8.7895, 9.6286, 9.6321, 8.7805, 9.629, 9.6255, 8.6699, 9.7913, 9.9848, 7.7804, 9.1515, 9.3437, 9.9911, 9.9651, 9.9473, 9.9911, 9.9545, 9.9418, 9.8284, 9.837, 9.8386, 9.8248, 9.8366, 9.8385, 9.9623, 9.9719, 9.9736, 9.9463, 9.9679, 9.9711, 7.5566, 8.0346, 7.93, 6.7018, 8.4457, 8.3838, 9.7845, 9.7261, 9.7845, 9.7261, 9.8897, 9.9975, 9.8367, 9.8866, 8.7421, 8.3307, 8.7376, 8.3312]
    log_cvvdp= []
    pu_cvvdp= [9.978, 9.978, 9.993, 9.979, 9.979, 9.6402, 9.2318, 8.85, 8.8397, 8.3248, 8.0821, 9.8201, 9.8997, 9.9, 9.8198, 9.7244, 9.7115, 9.6449, 9.7114, 9.6909, 9.4817, 9.6513, 9.6427, 9.99, 9.9959, 9.9939, 9.8658, 9.8042, 9.8042, 9.9812, 9.9874, 9.9915, 9.9856, 9.9918, 9.9929, 8.0122, 7.7785, 7.5117, 7.9466, 7.8899, 7.6095, 9.9842, 9.96, 9.9611, 9.6269, 9.6091, 9.6089, 8.2128, 7.7302, 7.6399, 7.6176, 6.7822, 6.6972, 8.0027, 9.6251, 9.661, 7.8966, 9.6423, 9.6939, 7.8266, 8.2949, 7.9877, 8.023, 8.9535, 8.7534, 7.5214, 8.3437, 8.2507, 6.9745, 7.7124, 7.6669, 8.9301, 8.1574, 8.0369, 9.3242, 9.0483, 9.0589, 8.7923, 9.6319, 9.6354, 8.7832, 9.6319, 9.6287, 8.693, 9.7913, 9.9778, 7.8202, 9.1648, 9.3576, 9.9911, 9.9643, 9.947, 9.9911, 9.9647, 9.9475, 9.7758, 9.8019, 9.8165, 9.7748, 9.802, 9.8162, 9.9623, 9.9718, 9.9735, 9.9463, 9.9677, 9.971, 7.6024, 8.0699, 7.9654, 6.7629, 8.483, 8.4212, 9.7845, 9.7261, 9.7845, 9.7261, 9.8897, 9.9975, 9.8375, 9.8866, 8.7419, 8.3219, 8.7373, 8.3224]
    pq_cvvdp= [9.9488, 9.9486, 9.9038, 9.911, 9.908, 9.6473, 9.2365, 8.8545, 8.8451, 8.3346, 8.0926, 9.935, 9.8614, 9.8637, 9.9218, 9.7208, 9.7083, 9.7495, 9.7365, 9.7157, 9.6623, 9.7159, 9.701, 9.99, 9.9959, 9.9939, 9.8862, 9.8249, 9.8219, 9.9812, 9.9874, 9.9915, 9.9856, 9.9918, 9.9929, 7.9569, 7.7248, 7.4637, 8.0029, 7.8997, 7.6234, 9.9842, 9.96, 9.9611, 9.6292, 9.6095, 9.6093, 8.3611, 7.6552, 7.4677, 7.8341, 6.925, 6.7808, 8.0395, 9.7023, 9.7142, 7.9293, 9.7574, 9.824, 7.6919, 8.1601, 7.804, 7.8366, 8.659, 8.3401, 7.4411, 8.336, 8.0741, 6.9893, 7.74, 7.6402, 8.8299, 8.1113, 7.9871, 9.3196, 9.0544, 9.0549, 8.8019, 9.6299, 9.6331, 8.7916, 9.63, 9.6268, 8.669, 9.79, 9.9713, 7.8614, 9.1386, 9.3119, 9.9911, 9.964, 9.9469, 9.9911, 9.9646, 9.9475, 9.7066, 9.6556, 9.7041, 9.7083, 9.6519, 9.6972, 9.9623, 9.9718, 9.9735, 9.9463, 9.9677, 9.971, 7.4168, 7.8424, 7.7688, 7.0424, 8.563, 8.4726, 9.7845, 9.7261, 9.7845, 9.7261, 9.9093, 9.9975, 9.8603, 9.9019, 8.7523, 8.447, 8.7476, 8.4467]

    # print(len(linear_psnr_rgb), len(pu_psnr_rgb), len(pq_psnr_rgb))

    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(linear_cvvdp, pu_cvvdp)
    print('t statistic:', t_statistic)
    print('p value:', p_value)

    navie_psnr_rgb = np.array(navie_psnr_rgb)
    linear_psnr_rgb = np.array(linear_psnr_rgb)
    pu_psnr_rgb = np.array(pu_psnr_rgb)
    pq_psnr_rgb = np.array(pq_psnr_rgb)
    print(np.mean(linear_psnr_rgb), np.mean(pu_psnr_rgb), np.mean(pq_psnr_rgb))
    psnr_rgb_df = pd.DataFrame({'linear_psnr_rgb': linear_psnr_rgb, 'pu_psnr_rgb': pu_psnr_rgb, 'pq_psnr_rgb': pq_psnr_rgb})


    navie_psnr_y = np.array(navie_psnr_y)
    linear_psnr_y = np.array(linear_psnr_y)
    pu_psnr_y = np.array(pu_psnr_y)
    pq_psnr_y = np.array(pq_psnr_y)
    print(np.mean(linear_psnr_y), np.mean(pu_psnr_y), np.mean(pq_psnr_y))
    psnr_y_df = pd.DataFrame({ 'linear_psnr_y': linear_psnr_y, 'pu_psnr_y': pu_psnr_y, 'pq_psnr_y': pq_psnr_y})

    navie_cvvdp = np.array(navie_cvvdp)
    linear_cvvdp = np.array(linear_cvvdp)
    pu_cvvdp = np.array(pu_cvvdp)
    pq_cvvdp = np.array(pq_cvvdp)
    print(np.mean(linear_cvvdp), np.mean(pu_cvvdp), np.mean(pq_cvvdp))
    cvvdp_df = pd.DataFrame({'linear_cvvdp': linear_cvvdp, 'pu_cvvdp': pu_cvvdp, 'pq_cvvdp': pq_cvvdp})


    sns.violinplot(data=psnr_rgb_df)
    plt.ylabel('dncnn_psnr_rgb')
    plt.savefig('dncnn_psnr_rgb.png')
    plt.clf()

    sns.violinplot(data=psnr_y_df)
    plt.ylabel('dncnn_psnr_y')
    plt.savefig('dncnn_psnr_y.png')
    plt.clf()

    sns.violinplot(data=cvvdp_df)
    plt.ylabel('dncnn_cvvdp')
    plt.savefig('dncnn_cvvdp.png')
    plt.clf()



"""
------------------- super-res -------------------
EDSR
t statistic: -6.083389863175833
p value: 6.6656545444680144e-09
49.23061058823528 39.90405764705882 49.296732941176465 49.83362235294118
60.63825294117646 48.269279999999995 60.97222823529411 61.861292941176465
9.459194623655915 8.832582795698924 9.47511075268817 9.536794623655911

WDSR
t statistic: -10.671725507566991
p value: 3.740696824621314e-22
49.16300873015873 35.870330952380954 46.611673015873016 49.08832857142857
60.58566904761904 40.18233174603175 57.61309523809524 60.78958015873015
9.410942857142855 8.453744444444444 9.32972857142857 9.480904761904764


------------------- denoise -------------------
DnCNN
t statistic: -0.0876360620942365
p value: 0.930236659476432
45.205743999999996 46.3487544 46.355635199999995
58.762662399999996 59.5669656 59.586996799999994
9.1246272 9.135019199999999 9.124940800000001
"""
