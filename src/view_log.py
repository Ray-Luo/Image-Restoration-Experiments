import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd
import math

def parse_tensorboard_logs(log_path):
    prev_epoch = 0
    data = []
    losses = []
    epoches = []
    cur_loss = 0
    for event in summary_iterator(log_path):
        for value in event.summary.value:
            cur_epoch = value.simple_value if value.tag == "epoch" else ""
            loss = value.simple_value if value.tag != "epoch" else 0

            if cur_epoch == "":
                cur_epoch = prev_epoch

            if cur_epoch == prev_epoch:
                cur_loss += abs(loss)
            else:
                losses.append(cur_loss)
                epoches.append(prev_epoch)
                cur_loss = 0
                prev_epoch = cur_epoch

            data.append({"wall_time": event.wall_time, "step": event.step, "name": value.tag, "value": value.simple_value})

    # losses.append(cur_loss)
    # epoches.append(prev_epoch)

    print(losses)
    print(epoches)
    print(len(losses), len(epoches))
    min_index = min(enumerate(losses), key=lambda x: x[1])[0]
    print(min_index)
    print(epoches[min_index], losses[min_index])
    return pd.DataFrame(data)

df = parse_tensorboard_logs('/home/luoleyouluole/Image-Restoration-Experiments/data/events.out.tfevents.1684434823.twshared28327.03.prn5.facebook.com.1780.0827c2d8a-3ad1-4bff-b52d-e2400b5c04b2')

# df.to_csv('log.csv', index=False) 143.32050898112357

# a=[6805.62321956642, 24073.124174844474, 2760.982276428491, 232.52980781020597, 211.61310666054487, 213.6679002828896, 192.8947196737863, 190.62209107354283, 191.64396223425865, 185.3434621989727, 185.98981623351574, 189.6610645353794, 171.23435285600135, 183.36131728813052, 164.04663955420256, 176.35859838873148, 171.47032137215137, 175.0026094429195, 180.20648355782032, 184.69739259034395, 178.42743533477187, 167.84214507974684, 166.81105288490653, 174.70882707461715, 170.26697011105716, 166.19137367233634, 169.54308485984802, 173.37649008631706, 152.16467777825892, 171.3496381584555, 165.16467525996268, 161.7536774147302, 161.89458448812366, 161.67116227000952, 168.42315449938178, 176.0053541213274, 163.57803223840892, 166.09436385519803, 165.18668721616268, 163.0765328295529, 166.85836979560554, 166.22397587634623, 164.79636185616255, 143.32050898112357, 166.24104135483503, 174.86345056071877, 163.53325148299336, 165.14732781611383, 164.8827736452222, 165.31170755252242, 160.80207073315978, 158.9360298588872, 162.6433554124087, 163.1564990915358, 161.35627206601202, 164.88970350660384, 158.77948692440987, 145.3501287791878, 159.04053460806608, 167.04614466428757, 156.37601733393967, 161.53002702631056, 161.17438263818622, 162.925024561584, 166.5996134467423, 159.54702974669635, 151.9416688401252, 161.86320586688817, 165.53423649817705, 167.78112800000235, 169.2537965103984, 144.74939426593482, 158.42912172153592, 154.07122202590108, 160.6564458515495, 163.2596446443349, 159.57626918703318, 160.6424630768597, 158.04170224815607, 159.94350340217352, 159.93445038050413, 158.16648927889764, 158.2974812462926, 158.2751636467874, 159.0386720597744, 146.14583315700293, 159.5725028384477, 162.1668068766594, 160.8698201328516, 163.03039931692183, 163.68159076943994, 159.01499747484922, 160.55929315462708, 163.07250429317355, 166.2911755014211, 160.71671854145825, 152.62939347326756, 163.34110496006906, 163.54577580466866, 142.59127073548734]
# b=[0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0]

# import numpy as np
# print(len(a), len(b))
# a = np.array(a)
# b = np.array(b)
# print(np.min(a))
# min_index = np.argmin(a)
# print(min_index, b[min_index])
