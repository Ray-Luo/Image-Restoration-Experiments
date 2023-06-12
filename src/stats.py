import scipy.stats as stats
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



# ********************************************* EDSR *********************************************
if 0:
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
    navie_cvvdp= [8.6396, 8.7445, 9.6188, 9.7385, 9.8731, 9.987, 8.2492, 9.2435, 9.2518, 9.4614, 9.3869, 9.0232, 8.6113, 9.4372, 9.276, 9.9706, 9.7047, 9.9648, 8.3687,]
    linear_cvvdp= [7.1142, 6.9423, 9.2767, 9.7136, 9.8607, 9.8907, 7.3931, 8.2909, 7.3717, 8.1232, 7.7942, 7.3115, 8.1655, 8.3715, 7.2113, 9.8725, 9.2715, 9.898, 7.0458, ]
    log_cvvdp= []
    pu_cvvdp= [8.7742, 9.0332, 9.7219, 9.7693, 9.8873, 9.987, 8.217, 9.4318, 9.3393, 9.5702, 9.4411, 9.2575, 8.5731, 9.6284, 9.4777, 9.9744, 9.8026, 9.9655, 8.4168, ]
    pq_cvvdp= [8.7803, 9.0715, 9.7442, 9.7692, 9.8873, 9.987, 8.2389, 9.4499, 9.3897, 9.6242, 9.4731, 9.3007, 8.5847, 9.6491, 9.5278, 9.9744, 9.8326, 9.9655, 8.4296,]

    # print(len(navie_psnr_rgb), len(linear_psnr_rgb), len(pu_psnr_rgb), len(pq_psnr_rgb  ))

    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(navie_psnr_y, pq_psnr_y)
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
    navie_cvvdp= [8.6396, 8.7445, 9.6188, 9.7385, 9.8731, 9.987, 8.2492, 9.2435, 9.2518, 9.4614, 9.3869, 9.0232, 8.6113, 9.4372, 9.276, 9.9706, 9.7047, 9.9648, 8.3687]
    linear_cvvdp= [7.2457, 7.1853, 9.2965, 9.7312, 9.7469, 9.9229, 7.4938, 8.4019, 7.5229, 8.3027, 7.8529, 7.4638, 8.2099, 8.4619, 7.3755, 9.7, 9.3184, 9.9224, 7.1509]
    log_cvvdp= []
    pu_cvvdp= [8.9043, 9.205, 9.7582, 9.7909, 9.8926, 9.987, 8.3793, 9.4795, 9.4221, 9.6375, 9.5028, 9.3001, 8.7167, 9.6689, 9.6303, 9.9791, 9.8449, 9.9674, 8.4915]
    pq_cvvdp= [8.9321, 9.2339, 9.7689, 9.7978, 9.894, 9.987, 8.3977, 9.5095, 9.4485, 9.6516, 9.5218, 9.3298, 8.7245, 9.6884, 9.6862, 9.9795, 9.87, 9.9672, 8.5107]

    # print(len(navie_psnr_rgb), len(linear_psnr_rgb), len(pu_psnr_rgb), len(pq_psnr_rgb  ))

    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(navie_psnr_rgb, pq_psnr_rgb)
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
if 0:
    navie_psnr_rgb= []
    linear_psnr_rgb= [45.3063, 41.145, 48.0028, 36.449, 44.8135, 62.9603, 30.1688, 59.8898, 28.1732, 31.1353, 28.3828, 28.766, 35.7284, 49.3166, 38.8066, 60.0217, 49.0434, 54.6459, 30.5237, 54.24, 53.0709, 39.5226]
    log_psnr_rg= []
    pu_psnr_rgb= [68.6411, 41.1965, 48.4424, 38.4425, 45.1201, 63.0932, 30.4402, 61.0794, 28.2675, 32.8559, 28.3881, 29.3291, 36.3005, 50.1633, 38.8077, 60.4938, 51.7034, 55.2396, 30.5533, 57.5505, 56.1019, 41.7249]
    pq_psnr_rgb= [64.7715, 41.237, 48.4149, 38.8939, 45.5637, 62.9739, 30.5816, 62.263, 29.7226, 33.0245, 28.6596, 29.7386, 36.6074, 50.3382, 38.8214, 60.4353, 52.6349, 55.1868, 31.2749, 63.4621, 58.9449, 47.1125]
    navie_psnr_y= []
    linear_psnr_y= [66.541, 46.7088, 68.3437, 59.3963, 67.761, 85.9045, 32.5945, 66.2019, 31.9422, 38.6396, 30.9281, 29.8528, 37.8299, 52.8527, 42.3161, 82.9653, 71.7756, 77.5855, 33.1932, 63.0398, 76.0186, 41.4595]
    log_psnr_y= []
    pu_psnr_y= [89.1848, 46.7097, 68.5786, 61.3897, 68.0676, 86.0373, 32.6324, 66.2527, 31.9653, 38.6827, 30.9293, 29.9297, 37.9152, 52.8863, 42.3162, 83.4369, 74.2615, 78.1779, 33.1994, 63.2814, 79.0497, 43.6199]
    pq_psnr_y= [70.5807, 46.7104, 68.5641, 61.8409, 68.5112, 85.9181, 32.6579, 66.2649, 32.3632, 38.7051, 30.9755, 30.0009, 37.9694, 52.9117, 42.3187, 83.3784, 75.089, 78.125, 33.3345, 69.9575, 81.8926, 48.8492]
    navie_cvvdp= []
    linear_cvvdp= [9.8953, 8.5985, 9.7998, 9.7237, 9.8732, 9.9864, 7.6133, 9.6684, 7.3907, 8.5295, 7.9677, 7.4293, 8.5637, 9.1585, 8.3968, 9.9654, 9.8815, 9.9542, 7.4043]
    log_cvvdp= []
    pu_cvvdp= [9.9886, 8.5985, 9.7997, 9.8232, 9.8882, 9.9865, 7.6501, 9.6684, 7.4002, 8.5333, 7.9679, 7.4569, 8.6151, 9.1634, 8.3968, 9.9645, 9.8996, 9.9599, 7.4063]
    pq_cvvdp= [9.8892, 8.5985, 9.7996, 9.8429, 9.8987, 9.987, 7.6621, 9.6685, 7.4825, 8.5386, 7.985, 7.4842, 8.64, 9.1665, 8.3973, 9.9635, 9.9032, 9.9607, 7.4388]

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
