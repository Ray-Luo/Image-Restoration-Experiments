import scipy.stats as stats
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



# ********************************************* EDSR *********************************************
if 1:
    navie_psnr_rgb= [42.3617, 42.2483, 46.8167, 37.2652, 45.1487, 63.409, 36.1996, 51.0323, 47.7487, 40.0868, 47.9425, 43.1203, 39.9199, 51.2866, 49.4526, 61.8666, 51.4706, 56.4004, 38.04, 54.1844, 53.6496, 40.1452]
    linear_psnr_rgb= [31.2766, 31.9791, 44.2531, 36.6029, 44.911, 60.6837, 27.335, 42.0781, 28.2908, 31.4776, 27.2305, 28.7293, 30.4772, 40.6892, 31.7934, 59.9276, 47.5458, 55.1204, 28.1383, 54.1844, 53.4545, 39.5661]
    log_psnr_rg= []
    pu_psnr_rgb= [42.6047, 42.934, 47.1148, 37.7485, 45.3128, 63.409, 36.2016, 52.5951, 48.0133, 40.1125, 47.8372, 44.3279, 40.2301, 52.14, 51.4325, 62.0177, 51.9473, 56.4319, 37.976, 54.1951, 54.8418, 40.4956]
    pq_psnr_rgb= [42.7952, 43.7031, 47.6296, 37.819, 45.3175, 63.409, 36.3704, 52.9543, 49.3245, 40.7018, 48.4952, 44.7345, 40.3636, 52.6168, 52.5558, 61.988, 52.1529, 56.4287, 38.2368, 54.179, 55.2472, 40.5454]
    navie_psnr_y= [44.3126, 44.6994, 60.2547, 60.2108, 68.0638, 86.2484, 40.537, 51.5516, 50.775, 54.4545, 50.8824, 48.2229, 43.4566, 54.8353, 51.3997, 84.2931, 63.3531, 79.0725, 40.7857, 62.9934, 76.5973, 41.9227]
    linear_psnr_y= [34.934, 34.9704, 52.5787, 58.6471, 67.0358, 71.7078, 28.8777, 42.3375, 31.6713, 35.5501, 29.1067, 29.4355, 31.2191, 41.7057, 33.1029, 71.5051, 52.1898, 72.1087, 30.2978, 62.9934, 76.4023, 41.4572]
    log_psnr_y= []
    pu_psnr_y= [44.8151, 46.5037, 62.3287, 60.6939, 68.2345, 86.2484, 40.4852, 53.5236, 51.7713, 55.7155, 51.3155, 50.5356, 43.419, 57.7395, 53.9429, 84.4303, 66.062, 79.1094, 40.9863, 63.074, 77.7895, 42.2403]
    pq_psnr_y= [44.9224, 47.0132, 63.2518, 60.7643, 68.2393, 86.2484, 40.6258, 53.9379, 52.6408, 56.7475, 51.6904, 51.2, 43.5137, 58.2836, 54.8899, 84.4138, 67.8375, 79.1068, 41.1053, 62.9537, 78.195, 42.286]
    navie_cvvdp= [8.6396, 8.7445, 9.6188, 9.7385, 9.8731, 9.987, 8.2492, 9.2435, 9.2518, 9.4614, 9.3869, 9.0232, 8.6113, 9.4372, 9.276, 9.9706, 9.7047, 9.9648, 8.3687, 9.7647, 9.8994, 8.6123]
    linear_cvvdp= [7.1142, 6.9423, 9.2767, 9.7136, 9.8607, 9.8907, 7.3931, 8.2909, 7.3717, 8.1232, 7.7942, 7.3115, 8.1655, 8.3715, 7.2113, 9.8725, 9.2715, 9.898, 7.0458, 9.7647, 9.8928, 8.5531]
    log_cvvdp= []
    pu_cvvdp= [8.7742, 9.0332, 9.7219, 9.7693, 9.8873, 9.987, 8.217, 9.4318, 9.3393, 9.5702, 9.4411, 9.2575, 8.5731, 9.6284, 9.4777, 9.9744, 9.8026, 9.9655, 8.4168, 9.7671, 9.9266, 8.6749]
    pq_cvvdp= [8.7803, 9.0715, 9.7442, 9.7692, 9.8873, 9.987, 8.2389, 9.4499, 9.3897, 9.6242, 9.4731, 9.3007, 8.5847, 9.6491, 9.5278, 9.9744, 9.8326, 9.9655, 8.4296, 9.7631, 9.9329, 8.6867]

    # print(len(navie_psnr_rgb), len(linear_psnr_rgb), len(pu_psnr_rgb), len(pq_psnr_rgb  ))

    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(linear_psnr_rgb, pu_psnr_rgb)
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
    navie_psnr_rgb= [42.3617, 42.2483, 46.8167, 37.2652, 45.1487, 63.409, 36.1996, 51.0323, 47.7487, 40.0868, 47.9425, 43.1203, 39.9199, 51.2866, 49.4526, 61.8666, 51.4706, 56.4004, 38.04, 54.1844, 53.6496, 40.1452]
    linear_psnr_rgb= [29.8302, 31.1368, 41.8324, 36.4386, 44.3173, 58.5204, 27.3148, 39.217, 27.9642, 30.8947, 27.1859, 28.7452, 30.1661, 38.3215, 30.8835, 49.9682, 45.2492, 47.5864, 27.5826, 54.1844, 53.4222, 39.5493]
    log_psnr_rg= []
    pu_psnr_rgb= [43.3785, 44.277, 47.7944, 38.1857, 45.4414, 63.415, 36.6188, 53.1005, 49.5656, 40.908, 48.4626, 44.6662, 40.7117, 52.897, 53.7129, 62.1999, 52.6282, 56.5086, 38.2928, 54.196, 55.194, 40.5511]
    pq_psnr_rgb= [43.5805, 44.7393, 48.079, 38.3108, 45.5003, 63.414, 36.7324, 53.5365, 50.2225, 41.202, 48.7072, 44.8821, 40.7564, 53.1575, 54.8778, 62.2976, 52.9623, 56.5007, 38.4215, 54.2247, 55.4299, 40.6047]
    navie_psnr_y= [44.3126, 44.6994, 60.2547, 60.2108, 68.0638, 86.2484, 40.537, 51.5516, 50.775, 54.4545, 50.8824, 48.2229, 43.4566, 54.8353, 51.3997, 84.2931, 63.3531, 79.0725, 40.7857, 62.9934, 76.5973, 41.9227]
    linear_psnr_y= [31.8063, 33.3353, 48.3736, 59.004, 58.6108, 74.3149, 29.2739, 40.549, 31.1279, 35.873, 29.2885, 29.677, 31.3527, 39.9235, 32.1557, 54.8454, 50.3683, 68.9336, 29.5655, 62.9934, 76.3699, 41.45]
    log_psnr_y= []
    pu_psnr_y= [45.5393, 47.8852, 63.4093, 61.1312, 68.3628, 86.2548, 40.9942, 54.1568, 52.8403, 56.7965, 51.7552, 51.0869, 43.9029, 58.4663, 56.159, 84.6509, 68.0282, 79.1931, 41.2502, 63.0807, 78.1417, 42.2701]
    pq_psnr_y= [45.7188, 48.2578, 63.9676, 61.2566, 68.4226, 86.2533, 41.0761, 54.6631, 53.3772, 57.259, 51.9729, 51.6229, 43.9459, 58.9025, 57.3163, 84.7378, 69.6137, 79.1843, 41.3605, 63.0819, 78.3776, 42.3207]
    navie_cvvdp= [8.6396, 8.7445, 9.6188, 9.7385, 9.8731, 9.987, 8.2492, 9.2435, 9.2518, 9.4614, 9.3869, 9.0232, 8.6113, 9.4372, 9.276, 9.9706, 9.7047, 9.9648, 8.3687, 9.7647, 9.8994, 8.6123]
    linear_cvvdp= [7.2457, 7.1853, 9.2965, 9.7312, 9.7469, 9.9229, 7.4938, 8.4019, 7.5229, 8.3027, 7.8529, 7.4638, 8.2099, 8.4619, 7.3755, 9.7, 9.3184, 9.9224, 7.1509, 9.7647, 9.8937, 8.5534]
    log_cvvdp= []
    pu_cvvdp= [8.9043, 9.205, 9.7582, 9.7909, 9.8926, 9.987, 8.3793, 9.4795, 9.4221, 9.6375, 9.5028, 9.3001, 8.7167, 9.6689, 9.6303, 9.9791, 9.8449, 9.9674, 8.4915, 9.7674, 9.9317, 8.675]
    pq_cvvdp= [8.9321, 9.2339, 9.7689, 9.7978, 9.894, 9.987, 8.3977, 9.5095, 9.4485, 9.6516, 9.5218, 9.3298, 8.7245, 9.6884, 9.6862, 9.9795, 9.87, 9.9672, 8.5107, 9.7674, 9.9345, 8.6829]

    # print(len(navie_psnr_rgb), len(linear_psnr_rgb), len(pu_psnr_rgb), len(pq_psnr_rgb  ))

    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(linear_psnr_rgb, pu_psnr_rgb)
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



# ********************************************* RealESRGAN *********************************************
if 1:
    navie_psnr_rgb= [42.3617, 42.816, 53.8016, 55.288, 60.0168, 61.5306, 42.2483, 42.2407, 46.6849, 47.4916, 50.2901, 52.2757, 46.8167, 43.9313, 40.588, 40.6285, 42.0127, 45.196, 37.2652, 36.7806, 37.2486, 37.5227, 38.4113, 39.955, 45.1487, 44.3106, 41.7271, 41.1735, 39.6754, 41.2265, 63.409, 60.2528, 48.2603, 46.2653, 40.7656, 39.4986, 36.1996, 36.5488, 39.2254, 39.5633, 40.2168, 41.0378, 51.0323, 49.0187, 47.0801, 47.924, 52.5065, 51.7093, 47.7487, 47.1917, 47.8156, 48.1163, 49.5583, 51.31, 40.0868, 40.6837, 42.1651, 42.4605, 43.4747, 44.7845, 47.9425, 48.0411, 47.9706, 48.1876, 49.2728, 50.3853, 43.1203, 42.9312, 45.3525, 45.895, 47.89, 48.7444, 39.9199, 40.6208, 42.056, 42.2708, 43.3913, 45.0462, 51.2866, 49.4203, 47.2311, 47.3582, 48.199, 49.7612, 49.4526, 47.3861, 45.5137, 45.5654, 46.3734, 48.0208, 61.8666, 58.554, 50.1315, 49.7427, 48.4168, 48.9315, 51.4706, 50.5314, 47.239, 46.7317, 46.1349, 48.1555, 56.4004, 52.5719, 43.5632, 43.1986, 42.6796, 43.6304, 38.04, 38.7614, 45.0069, 45.9365, 47.3965, 46.7491, 54.1844, 54.155, 54.1543, 54.1663, 54.1325, 54.0641, 53.6496, 53.5753, 53.7933, 53.9267, 54.4378, 55.5012, 40.1452, 40.0982, 40.0796, 40.086, 40.0818, 40.0927]
    linear_psnr_rgb= [35.9084, 31.5328, 25.7364, 25.6816, 26.5899, 25.7816, 35.4621, 32.9786, 26.7213, 26.4833, 26.6349, 26.5563, 44.2881, 41.1978, 34.4814, 33.7612, 29.4676, 25.4082, 36.1783, 35.6807, 34.6598, 34.4512, 32.4901, 30.7162, 44.3571, 43.8753, 39.0087, 37.7277, 32.9206, 29.9444, 62.24, 59.5416, 44.9456, 42.7279, 35.6664, 31.0618, 29.8777, 26.9481, 24.9998, 24.8015, 23.9125, 23.4706, 45.3874, 41.9311, 31.481, 30.5536, 28.0839, 26.7429, 30.2944, 27.2146, 24.0775, 23.8869, 23.1219, 22.7635, 31.4727, 30.9139, 27.2932, 26.761, 24.6635, 23.0217, 28.749, 26.6596, 23.914, 23.6365, 22.7578, 22.9189, 31.7848, 30.2811, 26.8796, 26.3882, 24.6672, 24.8023, 32.9926, 30.3087, 27.4607, 27.1301, 24.7255, 22.9416, 42.6149, 39.3515, 29.988, 29.0115, 26.1697, 25.1559, 34.0587, 31.5, 26.9082, 26.9013, 26.2087, 25.7485, 56.8048, 52.2268, 36.8692, 35.3901, 29.0003, 26.3032, 47.7506, 45.1192, 36.5384, 35.0994, 30.8317, 26.3852, 52.3972, 48.3771, 37.0091, 35.6507, 29.5795, 25.4285, 30.7483, 27.8188, 23.0777, 22.9915, 24.0257, 24.2976, 54.1799, 54.1347, 54.1533, 54.1644, 54.1301, 54.0572, 53.2169, 53.1426, 53.3289, 53.4341, 53.6444, 54.4954, 39.4519, 39.3884, 39.3525, 39.3683, 39.3768, 39.4059]
    log_psnr_rg= []
    pu_psnr_rgb= [42.3571, 42.465, 49.8108, 50.6575, 52.3357, 50.8161, 42.5045, 42.4462, 46.2708, 46.949, 48.7933, 48.6517, 47.0039, 44.4035, 40.959, 41.0674, 42.0068, 44.1324, 37.7053, 37.1922, 36.9963, 37.1965, 37.7045, 38.7052, 45.2455, 44.3873, 41.5876, 40.995, 39.3049, 40.7692, 63.1971, 60.2354, 48.3892, 46.5799, 41.5151, 40.3671, 35.2244, 35.7842, 37.7715, 37.955, 38.1238, 38.5988, 51.657, 49.5842, 45.505, 45.895, 48.3324, 47.7367, 47.5334, 47.1429, 46.6005, 46.671, 47.3039, 48.7876, 39.9105, 39.9039, 39.9247, 40.1162, 40.6671, 41.4689, 46.3496, 46.4818, 46.2386, 46.3871, 47.0928, 48.02, 43.4229, 42.2279, 38.447, 38.3153, 38.4233, 41.3243, 39.4296, 40.18, 41.3913, 41.4797, 41.9235, 42.5813, 51.6075, 49.895, 46.7422, 46.687, 46.3157, 47.1974, 51.5107, 49.5722, 46.2795, 46.3789, 46.9875, 47.5506, 61.6866, 58.7209, 49.762, 49.3155, 47.9724, 48.4364, 51.9509, 51.0354, 48.3211, 47.7356, 46.4861, 46.8537, 56.0941, 52.3948, 43.3854, 43.0582, 42.4055, 42.9116, 36.5336, 37.0349, 43.0428, 44.0358, 46.2276, 46.5773, 53.361, 53.8377, 54.1498, 54.161, 54.132, 54.0668, 54.6989, 54.6371, 54.0132, 54.0384, 54.8222, 55.513, 40.3492, 40.4347, 40.0654, 40.0126, 39.9014, 39.8763]
    pq_psnr_rgb= [42.5819, 42.6642, 52.7177, 54.0182, 55.5301, 53.4684, 43.1354, 42.9636, 46.7107, 47.5723, 50.4823, 50.9173, 47.2725, 44.9119, 41.2925, 41.3004, 42.0323, 44.383, 37.6961, 37.4569, 37.7387, 37.9524, 38.6859, 40.3101, 45.1024, 44.4767, 42.0312, 41.4407, 39.8174, 41.1214, 62.8896, 60.1851, 48.5784, 46.7572, 41.7381, 40.5153, 35.1555, 35.8951, 39.8929, 40.3425, 41.241, 42.1511, 52.0644, 50.0079, 46.15, 46.734, 50.4655, 47.8058, 48.8472, 48.6233, 48.6904, 48.7662, 49.4268, 50.3296, 40.3438, 41.0717, 42.8503, 43.1269, 43.8431, 44.8692, 47.0329, 47.1319, 47.2954, 47.4269, 48.2641, 49.198, 43.9775, 44.325, 45.7761, 46.1566, 46.962, 47.1845, 39.2215, 40.217, 42.554, 42.7423, 43.6887, 44.8534, 51.8467, 50.4391, 47.0679, 46.8422, 47.2248, 49.5435, 52.8195, 50.724, 47.6321, 47.8046, 47.9257, 48.4536, 60.8769, 58.8599, 50.4472, 49.9812, 48.8911, 49.5994, 52.1333, 51.573, 49.4578, 49.0294, 48.1527, 48.0744, 54.9496, 52.3415, 44.0128, 43.6374, 42.6328, 42.7859, 36.3206, 37.0871, 43.869, 45.0354, 47.5766, 47.6151, 52.6839, 53.0335, 54.1424, 54.1616, 54.1369, 54.0742, 54.6789, 54.8377, 55.0424, 55.1375, 55.8163, 56.8981, 40.4143, 40.4428, 40.4126, 40.4134, 40.3953, 40.3791]
    navie_psnr_y= [44.3126, 43.0664, 53.9278, 55.5225, 61.2981, 63.6394, 44.6994, 43.7891, 48.3696, 49.3435, 52.513, 54.3626, 60.2547, 55.1891, 43.8545, 43.0329, 42.2461, 45.1362, 60.2108, 59.7152, 57.5113, 56.2375, 50.0615, 45.6994, 68.0638, 66.1346, 52.4987, 51.3098, 47.3696, 44.4573, 86.2484, 82.8905, 64.5865, 60.4124, 46.7104, 41.2926, 40.537, 42.0332, 43.6047, 43.4614, 43.1609, 43.471, 51.5516, 49.5083, 47.0889, 47.9872, 53.3876, 52.5191, 50.775, 49.7101, 49.5887, 49.7164, 50.5458, 52.0156, 54.4545, 52.9696, 49.8492, 49.6125, 48.7653, 47.9318, 50.8824, 51.8786, 51.22, 51.2721, 51.9343, 52.4775, 48.2229, 47.5978, 49.7752, 50.2268, 52.4015, 52.562, 43.4566, 44.9634, 47.1986, 47.042, 46.5391, 47.234, 54.8353, 52.5373, 48.7546, 48.8461, 49.4824, 51.0274, 51.3997, 48.7073, 46.5, 46.5143, 47.0371, 48.6316, 84.2931, 74.4941, 53.2838, 52.2517, 49.7124, 49.5016, 63.3531, 59.4624, 49.9338, 49.1561, 47.5462, 49.111, 79.0725, 74.7129, 58.2156, 54.9117, 45.9449, 44.4374, 40.7857, 40.6782, 46.061, 47.1251, 49.0194, 47.9116, 62.9934, 62.9507, 62.9501, 62.9674, 62.9181, 62.8344, 76.5973, 76.523, 76.7627, 76.8926, 77.4174, 78.5131, 41.9227, 41.8569, 41.8264, 41.8335, 41.8211, 41.8102]
    linear_psnr_y= [40.2847, 36.3326, 26.9992, 26.4227, 27.2114, 27.0639, 39.4464, 36.2552, 28.4482, 27.883, 27.2671, 27.7397, 56.1249, 51.0437, 38.4568, 37.1399, 32.8944, 28.5574, 59.1194, 58.5421, 49.8421, 48.0127, 41.747, 36.7736, 67.2428, 63.9355, 46.1774, 44.2815, 37.98, 33.236, 85.0119, 82.0474, 54.3729, 50.3171, 40.1931, 34.5588, 32.4817, 29.0889, 27.1474, 26.8961, 25.4173, 24.5771, 47.2938, 43.7825, 34.6417, 33.1115, 28.9411, 27.4802, 36.6411, 31.8517, 25.5634, 25.3709, 24.5425, 23.5652, 38.7454, 36.1972, 32.6789, 32.2486, 28.7093, 25.5194, 32.4611, 29.5389, 25.5413, 25.3002, 23.6241, 23.409, 34.9026, 32.9155, 29.1691, 28.6139, 25.5997, 25.6083, 34.7737, 31.7592, 28.9725, 28.9115, 26.7097, 24.3253, 44.9874, 41.8184, 32.7632, 31.7104, 27.6191, 25.8963, 36.9823, 33.83, 27.8354, 27.8115, 26.7782, 26.532, 72.6157, 57.4188, 40.0097, 38.539, 32.1724, 27.4982, 54.5013, 50.6502, 41.3258, 39.6262, 34.4778, 30.4845, 74.908, 70.2482, 47.4197, 44.4172, 36.7603, 29.2605, 33.8984, 30.4506, 24.2819, 23.9139, 24.7264, 24.8531, 62.9488, 62.9342, 62.9434, 62.9529, 62.9295, 62.8053, 76.1497, 75.9574, 76.2858, 76.3866, 76.6139, 77.4821, 41.387, 41.312, 41.2667, 41.2881, 41.2674, 41.2289]
    log_psnr_y= []
    pu_psnr_y= [44.9806, 43.1647, 51.242, 51.934, 53.9571, 54.937, 46.7767, 45.4873, 48.2769, 49.1471, 51.8095, 52.6743, 62.3842, 57.3534, 46.0059, 44.9458, 43.0588, 45.2726, 60.651, 60.1281, 58.1399, 57.1784, 51.5852, 47.133, 68.1557, 66.4982, 54.6745, 53.2607, 48.4991, 45.1125, 86.0442, 82.8793, 64.787, 60.8122, 48.3212, 42.8897, 39.3562, 40.9063, 43.5316, 43.4473, 43.0575, 42.8773, 52.9288, 50.8505, 46.1177, 46.6076, 49.9096, 49.3986, 51.1348, 50.8921, 50.3668, 50.2542, 49.8451, 50.8052, 54.8789, 53.963, 50.5537, 50.3686, 49.2659, 48.1621, 49.7757, 50.917, 49.5321, 49.5882, 50.3437, 51.0216, 49.7017, 48.9121, 46.7559, 46.7211, 46.6937, 48.3008, 42.2869, 43.5973, 47.136, 46.9729, 46.3507, 46.3814, 57.066, 54.8681, 49.1576, 48.9808, 48.3098, 49.4775, 54.6348, 52.0578, 47.8398, 47.9292, 48.2776, 49.0395, 84.0738, 74.2524, 53.7512, 52.4845, 50.2047, 49.9815, 66.7369, 62.7451, 52.7676, 52.2685, 49.8712, 49.6109, 78.7706, 74.4959, 59.0732, 55.8785, 46.5506, 44.5062, 39.6075, 39.3781, 44.8542, 46.1116, 48.7129, 48.5805, 60.6862, 62.0851, 62.9107, 62.9338, 62.9145, 62.8561, 74.9874, 77.5848, 77.0049, 77.0343, 77.8547, 78.4962, 42.3238, 42.209, 41.8695, 41.8302, 41.7444, 41.7136]
    pq_psnr_y= [44.8354, 43.0121, 53.1148, 54.5725, 57.1206, 55.5931, 46.8513, 45.2041, 48.0293, 49.0559, 52.8386, 53.1008, 62.7762, 57.8727, 45.7756, 44.5799, 42.4357, 44.4576, 60.6419, 60.3915, 58.7288, 57.7548, 52.1533, 47.1758, 68.0175, 66.621, 54.9715, 53.5074, 48.5354, 44.7155, 85.7364, 82.8257, 64.676, 60.6991, 48.2646, 42.7784, 38.6226, 40.3334, 44.5989, 44.6634, 44.8243, 45.3472, 53.296, 51.1584, 46.5619, 47.1756, 51.7743, 49.1377, 51.8772, 51.46, 50.7582, 50.7074, 50.8646, 51.5882, 55.633, 54.7774, 51.478, 51.127, 49.324, 48.1969, 50.1619, 50.953, 49.9378, 49.952, 50.8931, 51.743, 50.4639, 50.5974, 50.7291, 51.0341, 51.32, 50.7194, 41.465, 43.154, 47.9748, 47.8529, 47.1811, 47.5645, 57.5628, 55.3669, 49.2587, 48.4748, 48.6923, 51.3208, 55.3263, 52.7146, 49.083, 49.2342, 48.9163, 49.2905, 83.3797, 74.2251, 53.8652, 52.5829, 50.5402, 50.5676, 68.2509, 63.9728, 53.5041, 52.9092, 50.3423, 49.6221, 77.6553, 74.5047, 58.9875, 55.4374, 45.7825, 43.5838, 38.8394, 38.9983, 44.9446, 46.2819, 49.588, 49.4797, 58.2146, 60.2653, 62.9069, 62.9376, 62.9454, 62.9058, 70.7405, 76.1124, 78.0307, 78.1198, 78.8276, 79.8422, 42.4641, 42.3261, 42.1381, 42.1364, 42.1082, 42.0599]
    navie_cvvdp= [8.6396, 8.4692, 9.3419, 9.438, 9.7054, 9.8268, 8.7445, 8.6576, 8.9471, 9.0244, 9.2943, 9.4187, 9.6188, 9.4225, 8.6159, 8.5106, 8.3426, 8.6378, 9.7385, 9.7238, 9.6212, 9.5505, 9.1798, 8.7953, 9.8731, 9.815, 9.3077, 9.2408, 8.9106, 8.5566, 9.987, 9.9799, 9.7582, 9.639, 8.8337, 8.2402, 8.2492, 8.3945, 8.6481, 8.6259, 8.6199, 8.6685, 9.2435, 9.123, 8.874, 8.9522, 9.3759, 9.3607, 9.2518, 9.1576, 9.1662, 9.1803, 9.2282, 9.2941, 9.4614, 9.3696, 9.1929, 9.1777, 9.125, 9.0578, 9.3869, 9.441, 9.2562, 9.2532, 9.2801, 9.3315, 9.0232, 8.9764, 9.1448, 9.1676, 9.3003, 9.3215, 8.6113, 8.7484, 9.067, 9.0465, 8.9799, 9.04, 9.4372, 9.3215, 9.1016, 9.1149, 9.1922, 9.3108, 9.276, 9.0954, 8.8871, 8.8971, 8.9664, 9.0766, 9.9706, 9.9176, 9.3825, 9.3159, 9.1682, 9.1454, 9.7047, 9.5901, 9.1024, 9.0562, 8.9974, 9.1105, 9.9648, 9.9483, 9.5971, 9.4513, 8.8484, 8.6752, 8.3687, 8.3353, 8.8557, 8.9594, 9.13, 9.0222, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    linear_cvvdp= [8.2577, 7.5923, 7.3511, 7.4445, 8.1362, 8.0041, 8.2345, 7.7448, 6.523, 6.7526, 7.4965, 7.3127, 9.4527, 9.2053, 8.0286, 7.7845, 6.7681, 5.9706, 9.7089, 9.674, 9.135, 9.0062, 8.4366, 7.731, 9.873, 9.738, 8.9053, 8.729, 7.9391, 7.1442, 9.9864, 9.9777, 9.3014, 9.0337, 8.1104, 7.3281, 7.3543, 7.2521, 6.4802, 6.3542, 6.0276, 5.8424, 9.0141, 8.6646, 7.1438, 6.8429, 7.1463, 7.7525, 7.9993, 7.4191, 6.7049, 6.5906, 6.6047, 6.806, 8.495, 8.152, 7.4335, 7.343, 6.8191, 6.6921, 8.0128, 7.7441, 7.3634, 7.2486, 7.0863, 7.2412, 7.9492, 7.545, 6.7433, 6.6766, 6.5345, 6.8396, 7.8749, 7.7846, 7.2026, 7.0431, 6.1027, 5.6698, 8.6732, 8.4026, 6.9591, 6.764, 6.291, 6.5613, 7.9504, 7.444, 6.7252, 6.6813, 6.6839, 6.9677, 9.8169, 9.4193, 8.1936, 8.0244, 6.9895, 6.6715, 9.3821, 9.2112, 8.3314, 8.1062, 7.4094, 6.7267, 9.9534, 9.8978, 8.8854, 8.6066, 7.7348, 6.182, 7.4819, 7.0779, 6.611, 6.8004, 7.1944, 7.3976, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    log_cvvdp= []
    pu_cvvdp= [8.8589, 8.6626, 9.3309, 9.3701, 9.1813, 9.0608, 9.1016, 9.0422, 9.1726, 9.2225, 9.3253, 9.1233, 9.717, 9.568, 8.9686, 8.8695, 8.6701, 8.857, 9.8068, 9.7981, 9.7225, 9.6786, 9.4241, 9.1567, 9.8959, 9.8613, 9.5215, 9.4578, 9.1535, 8.8143, 9.9872, 9.9807, 9.7765, 9.6817, 9.1225, 8.6937, 8.2321, 8.4106, 8.832, 8.8179, 8.7632, 8.7474, 9.405, 9.3172, 8.9562, 8.9997, 9.3356, 9.169, 9.3572, 9.3776, 9.3564, 9.295, 9.0472, 9.1442, 9.5644, 9.516, 9.3609, 9.3512, 9.271, 9.2025, 9.4071, 9.4994, 9.1835, 9.1479, 9.0664, 8.9887, 9.2529, 9.1831, 9.1399, 9.1295, 9.0157, 8.886, 8.5964, 8.7474, 9.1215, 9.1021, 9.0557, 9.0483, 9.6182, 9.5611, 9.3321, 9.3338, 9.2945, 9.29, 9.6011, 9.5228, 9.1469, 9.1525, 9.1837, 9.118, 9.9792, 9.93, 9.5052, 9.437, 9.379, 9.3877, 9.8263, 9.741, 9.4157, 9.4096, 9.3497, 9.3496, 9.9669, 9.9507, 9.7141, 9.6077, 9.1036, 8.9494, 8.3106, 8.2636, 8.8727, 8.9609, 8.9957, 8.9075, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    pq_cvvdp= [8.8362, 8.636, 9.3468, 9.3894, 9.5108, 9.6081, 9.1072, 9.0273, 9.1652, 9.2057, 9.1887, 9.1961, 9.7319, 9.5879, 8.9599, 8.8524, 8.6367, 8.847, 9.8006, 9.7996, 9.7335, 9.6943, 9.4615, 9.1791, 9.8961, 9.8674, 9.5414, 9.4753, 9.1542, 8.7903, 9.9869, 9.981, 9.7697, 9.6751, 9.1206, 8.6936, 8.2574, 8.4524, 8.9745, 8.981, 8.9971, 9.0423, 9.4235, 9.3321, 9.0156, 9.0667, 9.4245, 9.3672, 9.4005, 9.4163, 9.4946, 9.5046, 9.4915, 9.4595, 9.5996, 9.5561, 9.4174, 9.4068, 9.3384, 9.2665, 9.4219, 9.5095, 9.3364, 9.3396, 9.3852, 9.435, 9.2936, 9.2995, 9.3405, 9.3577, 9.3518, 9.2906, 8.6158, 8.778, 9.2427, 9.2291, 9.1913, 9.2507, 9.6375, 9.5791, 9.3426, 9.3489, 9.3292, 9.4315, 9.6355, 9.5548, 9.2104, 9.232, 9.2764, 9.3111, 9.978, 9.9274, 9.4953, 9.4235, 9.3813, 9.4425, 9.8513, 9.7531, 9.4492, 9.4344, 9.3766, 9.3864, 9.9649, 9.9534, 9.711, 9.5953, 9.0717, 8.9491, 8.3057, 8.2495, 8.8632, 8.9813, 9.2074, 9.2478, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


    # print(len(linear_psnr_rgb), len(pu_psnr_rgb), len(pq_psnr_rgb))

    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(linear_psnr_rgb, pq_psnr_rgb)
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
t statistic: -2.7234627089745396
p value: 0.009370920415052452
39.80657272727273 47.7236090909091 48.07128636363637
57.22375 46.810663636363635 58.22564545454546 58.63032272727273
9.310327272727273 8.505886363636364 9.392586363636363 9.411927272727272

WDSR
t statistic: -3.7769777754946885
p value: 0.0004936292060938952
47.2634409090909 38.19595 48.30481363636363 48.55180454545455
57.22375 44.96328181818182 58.87979999999998 59.2131409090909
9.310327272727273 8.568968181818182 9.451463636363638 9.46745

RealESRGAN
t statistic: -12.524338665000478
p value: 1.6200217672550136e-28
34.100766666666665 45.60902803030303 46.67814393939394
39.8238840909091 52.97402878787879 53.38311666666667
6.662299242424243 8.00230909090909 8.064369696969697

"""
