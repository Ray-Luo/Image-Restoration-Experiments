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



# ********************************************* DnCNN *********************************************
if 1:
    navie_psnr_rgb= []
    linear_psnr_rgb= [42.3824, 40.1318, 39.8025, 38.9962, 38.3296, 52.308, 40.7266, 38.4851, 38.1962, 37.5453, 36.7468, 41.1742, 46.8378, 41.5966, 41.552, 39.5523, 38.3179, 48.2088, 35.3434, 35.4863, 35.5461, 34.8416, 34.6853, 35.8713, 43.9228, 41.8861, 41.5781, 40.3978, 38.4794, 44.7377, 60.0706, 46.2871, 44.7776, 40.7452, 39.7875, 63.0885, 29.8519, 29.6115, 30.009, 29.5691, 29.4249, 30.3339, 55.4596, 41.64, 42.099, 40.4529, 38.8062, 59.8098, 28.077, 27.9998, 27.9929, 27.8734, 27.8462, 28.1733, 30.4919, 30.2353, 30.4093, 30.0308, 29.9521, 30.958, 28.3193, 28.2791, 28.1354, 28.0841, 28.1032, 28.3841, 28.7392, 28.7159, 28.5275, 28.5014, 28.4331, 28.7666, 35.6374, 34.3318, 34.2712, 34.0248, 34.0036, 35.8164, 48.7494, 40.0423, 41.0659, 39.2633, 38.5239, 49.8194, 38.7791, 36.65, 36.5796, 36.3956, 35.9641, 38.8074, 60.0286, 42.9937, 46.3714, 41.1359, 39.9867, 60.4565, 48.7264, 44.0955, 42.3166, 39.3015, 38.6646, 49.4598, 52.9992, 41.5607, 41.3936, 39.6297, 39.6482, 55.2341, 30.4698, 30.1689, 30.0486, 30.0422, 29.982, 30.5285, 54.1515, 54.1515, 54.1515, 54.1276, 52.1323, 54.1844, 52.9276, 52.9283, 52.9276, 52.9383, 50.7945, 53.0468, 39.5888, 39.4268, 39.3896, 39.3591, 38.7853, 39.5827]
    log_psnr_rg= []
    pu_psnr_rgb= [64.9933, 63.4568, 64.0025, 66.6245, 67.8076, 68.1972, 41.0545, 40.9501, 40.9607, 41.0796, 41.1015, 41.1855, 47.2187, 47.9296, 47.9998, 46.8959, 48.06, 48.6611, 37.5979, 35.7231, 35.8194, 35.3344, 35.7703, 37.9847, 44.0386, 43.9222, 44.7207, 43.9201, 44.7271, 44.9225, 60.073, 58.3962, 57.9075, 59.445, 59.624, 63.0932, 29.9855, 29.8835, 30.3576, 29.9424, 29.8243, 30.4748, 60.1833, 58.13, 58.2864, 59.6588, 60.3894, 60.6136, 28.0712, 28.0195, 27.9937, 28.0855, 28.0678, 28.1733, 31.9197, 30.7154, 30.9272, 30.4754, 30.383, 32.8849, 28.3123, 28.2842, 28.334, 28.3209, 28.3487, 28.3834, 29.1221, 28.825, 28.7607, 28.7346, 28.766, 29.1876, 35.9807, 35.7451, 35.8277, 35.7321, 35.8002, 36.1502, 49.3355, 49.1511, 49.8308, 49.1861, 49.9721, 50.1361, 38.7743, 38.7589, 38.7447, 38.7871, 38.7332, 38.8074, 58.4526, 56.1031, 57.8046, 59.0736, 60.3397, 60.5066, 50.6793, 49.3689, 49.9443, 49.504, 49.4773, 50.6353, 53.0317, 55.1437, 55.0929, 55.1917, 55.1908, 55.2396, 30.4697, 30.4193, 30.4475, 30.4448, 30.4165, 30.5285, 54.335, 54.2328, 54.2202, 54.0586, 54.1215, 54.4663, 54.2355, 52.9618, 52.9243, 52.9243, 52.8963, 55.2724, 40.5257, 40.0089, 39.8009, 39.3862, 39.3412, 40.5868]
    pq_psnr_rgb= [52.2798, 61.7713, 63.7199, 66.8749, 68.0051, 67.9768, 40.9849, 40.512, 40.7807, 41.0556, 41.1014, 41.1859, 47.3159, 46.6277, 46.7114, 46.7944, 48.0618, 48.6585, 37.0929, 35.5088, 35.5972, 35.2809, 35.7703, 37.4916, 43.9798, 43.8445, 44.5746, 43.6488, 44.7229, 44.8237, 60.073, 51.4179, 50.9021, 55.8854, 59.2975, 63.0932, 29.9496, 29.8849, 30.362, 29.9413, 29.8242, 30.4983, 59.7444, 53.7191, 52.3168, 59.5361, 60.4299, 60.6294, 28.0712, 28.0195, 27.9937, 28.0842, 28.0658, 28.1977, 31.9371, 30.8627, 30.8971, 30.4359, 30.3634, 32.7241, 28.3124, 28.2839, 28.3252, 28.3094, 28.3473, 28.3855, 29.1694, 28.949, 28.7593, 28.7424, 28.7654, 29.248, 35.9752, 35.7925, 35.8798, 35.7414, 35.7987, 36.1646, 49.3396, 48.5493, 49.634, 48.5796, 49.9757, 50.1238, 38.7743, 38.6221, 38.6162, 38.7362, 38.7303, 38.8446, 58.9393, 46.623, 53.2883, 55.2014, 60.3452, 60.5427, 50.7444, 46.6315, 48.4179, 49.5059, 49.4799, 50.7034, 53.0243, 52.8911, 52.5036, 54.0205, 54.0083, 55.2403, 30.4808, 30.4123, 30.4343, 30.4313, 30.4099, 30.5414, 54.5064, 54.4558, 53.4252, 46.395, 52.5346, 54.8232, 53.3443, 52.9719, 52.4818, 50.9001, 50.6753, 54.1365, 40.7589, 40.3589, 40.0465, 39.4026, 39.3192, 40.8513]
    navie_psnr_y= []
    linear_psnr_y= [55.827, 48.9746, 48.6427, 47.7421, 46.876, 64.6137, 46.5717, 45.6677, 45.282, 44.904, 44.082, 46.688, 67.213, 55.0109, 54.8631, 50.2998, 47.5944, 68.4508, 58.2908, 58.1571, 58.0778, 55.7558, 51.1621, 58.8187, 66.8693, 61.9114, 59.7683, 57.1874, 49.5011, 67.6852, 83.0131, 66.8259, 63.1992, 52.2706, 49.9922, 86.0326, 32.5549, 32.5372, 32.5974, 32.5293, 32.4465, 32.6, 63.9739, 50.6509, 51.5293, 48.7881, 47.0661, 64.8312, 31.9058, 31.888, 31.8916, 31.8587, 31.8139, 31.9422, 38.6082, 38.5407, 38.5345, 38.4267, 38.3386, 38.6368, 30.903, 30.8953, 30.9225, 30.8717, 30.8769, 30.9284, 29.85, 29.8399, 29.8441, 29.8443, 29.8119, 29.8528, 37.7891, 37.672, 37.6135, 37.4279, 37.3629, 37.8274, 52.6851, 48.8317, 50.3197, 47.4327, 46.1497, 52.8092, 42.3168, 41.7301, 41.6172, 41.4718, 41.1835, 42.3162, 82.3655, 57.2524, 63.5889, 50.8255, 49.2241, 76.6184, 71.4645, 61.4989, 56.465, 49.556, 48.1723, 72.1633, 75.9314, 56.8492, 55.943, 49.7026, 49.5889, 78.1724, 33.1887, 33.1315, 33.0995, 33.1071, 33.0447, 33.1933, 62.9484, 62.9484, 62.9484, 62.9164, 61.011, 62.9934, 75.8753, 75.876, 75.8753, 75.886, 65.5829, 75.9945, 41.5935, 41.3675, 41.3229, 41.2853, 41.1983, 41.5673]
    log_psnr_y= []
    pu_psnr_y= [87.9381, 86.3075, 86.8905, 89.5131, 90.7143, 91.1436, 46.6992, 46.8034, 46.698, 46.8071, 46.6993, 46.7095, 67.9092, 68.2978, 68.3159, 67.6964, 68.3572, 68.6916, 60.5451, 58.6705, 58.7668, 58.2819, 58.7177, 60.9319, 66.9862, 66.8698, 67.6682, 66.8677, 67.6746, 67.8701, 83.0154, 81.3407, 80.8516, 82.3893, 82.5663, 86.0373, 32.5681, 32.545, 32.6178, 32.5936, 32.5342, 32.6211, 66.1411, 66.2284, 66.1949, 66.2228, 66.2376, 66.2445, 31.9038, 31.8909, 31.8943, 31.9083, 31.8951, 31.9422, 38.6386, 38.5618, 38.6394, 38.6098, 38.5973, 38.6828, 30.9, 30.8951, 30.9331, 30.9232, 30.9364, 30.9281, 29.9018, 29.8544, 29.8522, 29.8507, 29.8558, 29.9105, 37.8507, 37.7933, 37.8295, 37.7928, 37.7872, 37.8906, 52.8533, 52.8748, 52.8685, 52.8354, 52.8332, 52.88, 42.3297, 42.331, 42.2785, 42.3547, 42.2804, 42.3162, 81.3976, 79.0493, 80.7489, 82.0182, 83.2806, 83.4497, 73.3029, 72.0755, 72.6256, 72.2024, 72.18, 73.2754, 75.9652, 78.0818, 78.0298, 78.1301, 78.1249, 78.1779, 33.1887, 33.159, 33.1627, 33.1731, 33.1582, 33.1933, 62.9553, 62.9525, 62.952, 62.9448, 62.9472, 63.004, 77.1832, 75.9095, 75.872, 75.872, 75.8441, 78.2201, 42.1421, 41.6052, 41.4559, 41.2792, 41.2638, 42.2026]
    pq_psnr_y= [75.2274, 84.653, 86.6121, 89.7615, 90.9165, 90.9232, 46.6979, 46.7944, 46.6946, 46.8067, 46.6993, 46.7095, 67.9668, 67.5313, 67.5662, 67.6342, 68.3582, 68.6903, 60.0402, 58.4559, 58.5433, 58.2278, 58.7177, 60.4389, 66.9274, 66.7921, 67.5221, 66.5898, 67.6704, 67.7712, 83.0154, 74.365, 73.8491, 78.8316, 82.2402, 86.0373, 32.5773, 32.5633, 32.6335, 32.596, 32.5342, 32.6223, 66.1318, 65.4885, 64.863, 66.2189, 66.2384, 66.2448, 31.9038, 31.8909, 31.8943, 31.9083, 31.8951, 31.9483, 38.639, 38.5663, 38.6385, 38.6085, 38.5967, 38.6796, 30.9, 30.8951, 30.933, 30.9231, 30.9364, 30.9285, 29.9081, 29.8717, 29.8529, 29.8528, 29.8558, 29.9193, 37.8745, 37.8242, 37.8545, 37.7987, 37.7873, 37.9122, 52.8557, 52.8678, 52.8673, 52.8301, 52.8332, 52.8839, 42.3297, 42.3293, 42.277, 42.3541, 42.2804, 42.3231, 81.8839, 69.4153, 74.7706, 78.1478, 83.2869, 83.4857, 73.3633, 69.4492, 71.1765, 72.2042, 72.1825, 73.3388, 75.9579, 75.8331, 75.4454, 76.9612, 76.9457, 78.1787, 33.1909, 33.1589, 33.1626, 33.1729, 33.1582, 33.196, 62.9625, 62.9607, 62.9201, 62.1898, 62.8737, 63.0184, 76.2921, 75.9196, 75.4295, 73.8479, 73.6231, 77.0842, 42.2154, 41.9564, 41.8347, 41.6035, 41.2637, 42.2711]
    navie_cvvdp= []
    linear_cvvdp= [9.5499, 9.3083, 9.2915, 9.2393, 9.1831, 9.7426, 8.5927, 8.5878, 8.5727, 8.573, 8.5424, 8.5987, 9.7776, 9.5747, 9.5712, 9.3846, 9.2284, 9.7924, 9.5741, 9.5744, 9.5837, 9.5458, 9.4244, 9.5993, 9.8313, 9.7596, 9.7162, 9.6541, 9.347, 9.8493, 9.9795, 9.8731, 9.8068, 9.4835, 9.3718, 9.9865, 7.6027, 7.6007, 7.6084, 7.5961, 7.5852, 7.6189, 9.6436, 9.3756, 9.4103, 9.2895, 9.1852, 9.6522, 7.3908, 7.3888, 7.3883, 7.3944, 7.3906, 7.3907, 8.5303, 8.5268, 8.5284, 8.5262, 8.5218, 8.5298, 7.9647, 7.9599, 7.9652, 7.9614, 7.9617, 7.9675, 7.4275, 7.424, 7.4274, 7.4274, 7.4279, 7.4294, 8.547, 8.529, 8.5158, 8.4859, 8.4748, 8.5572, 9.1525, 9.0943, 9.12, 9.0559, 9.0054, 9.1572, 8.3983, 8.391, 8.3847, 8.389, 8.3752, 8.3968, 9.9603, 9.6401, 9.7636, 9.4031, 9.3271, 9.8995, 9.8392, 9.7553, 9.6356, 9.3495, 9.2686, 9.8497, 9.95, 9.6471, 9.6169, 9.3545, 9.3465, 9.96, 7.4054, 7.3972, 7.3991, 7.3993, 7.3964, 7.4043, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    log_cvvdp= []
    pu_cvvdp= [9.9859, 9.9755, 9.9786, 9.9759, 9.987, 9.9872, 8.5979, 8.608, 8.598, 8.6081, 8.5977, 8.5985, 9.7959, 9.79, 9.7925, 9.7841, 9.7915, 9.7997, 9.7711, 9.6115, 9.6133, 9.5756, 9.5877, 9.7837, 9.8547, 9.8346, 9.8501, 9.832, 9.8474, 9.8748, 9.9795, 9.9766, 9.9756, 9.9785, 9.9786, 9.9865, 7.6298, 7.6014, 7.6092, 7.6007, 7.5966, 7.6459, 9.6663, 9.6696, 9.6686, 9.6684, 9.6683, 9.6684, 7.3911, 7.3905, 7.3912, 7.3916, 7.3894, 7.3907, 8.5317, 8.5261, 8.531, 8.5312, 8.5317, 8.533, 7.9655, 7.9657, 7.9695, 7.9681, 7.9696, 7.9677, 7.4255, 7.4143, 7.4291, 7.4283, 7.4299, 7.4321, 8.577, 8.5574, 8.5579, 8.5509, 8.5516, 8.5879, 9.1601, 9.1602, 9.1604, 9.1572, 9.1565, 9.162, 8.3984, 8.3987, 8.3929, 8.4011, 8.3933, 8.3968, 9.9423, 9.9508, 9.9487, 9.9636, 9.9645, 9.9645, 9.8824, 9.8512, 9.856, 9.8492, 9.8493, 9.8767, 9.9501, 9.9599, 9.9598, 9.9599, 9.9599, 9.96, 7.4053, 7.4022, 7.402, 7.4037, 7.402, 7.4042, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    pq_cvvdp= [9.9678, 9.9758, 9.9817, 9.9839, 9.9867, 9.9872, 8.5979, 8.608, 8.598, 8.6081, 8.5977, 8.5985, 9.7958, 9.7894, 9.7922, 9.7841, 9.7915, 9.7986, 9.7181, 9.6067, 9.6101, 9.5745, 9.5876, 9.7315, 9.8441, 9.8341, 9.8494, 9.8318, 9.8474, 9.8631, 9.9795, 9.9595, 9.9578, 9.9745, 9.9784, 9.9865, 7.6445, 7.6306, 7.636, 7.6068, 7.5966, 7.6511, 9.6663, 9.6628, 9.6508, 9.6684, 9.6683, 9.6684, 7.3911, 7.3905, 7.3912, 7.3916, 7.3894, 7.3929, 8.5317, 8.5248, 8.5306, 8.5311, 8.5317, 8.5325, 7.9655, 7.9657, 7.9695, 7.9681, 7.9696, 7.9678, 7.4312, 7.4048, 7.414, 7.4151, 7.4299, 7.4415, 8.5916, 8.5722, 8.5725, 8.5563, 8.5517, 8.6022, 9.1613, 9.1617, 9.1612, 9.1591, 9.1565, 9.1636, 8.3984, 8.3987, 8.3929, 8.4011, 8.3933, 8.3982, 9.9384, 9.905, 9.8778, 9.9621, 9.9645, 9.9646, 9.8791, 9.8506, 9.8556, 9.8492, 9.8493, 9.874, 9.9501, 9.9574, 9.9569, 9.959, 9.9588, 9.96, 7.406, 7.4021, 7.402, 7.4036, 7.402, 7.4049, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


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
47.2634409090909 39.80657272727273 47.7236090909091 48.07128636363637
57.22375 46.810663636363635 58.22564545454546 58.63032272727273
9.310327272727273 8.505886363636364 9.392586363636363 9.411927272727272

WDSR
t statistic: -3.7769777754946885
p value: 0.0004936292060938952
47.2634409090909 38.19595 48.30481363636363 48.55180454545455
57.22375 44.96328181818182 58.87979999999998 59.2131409090909
9.310327272727273 8.568968181818182 9.451463636363638 9.46745


------------------- denoise -------------------
DnCNN
t statistic: -3.0213045866740336
p value: 0.002765893826202087
39.50819166666667 43.96426136363636 43.254508333333334
49.17157424242425 55.56540606060606 55.002667424242425
7.612794696969698 7.729848484848486 7.728222727272728
"""
