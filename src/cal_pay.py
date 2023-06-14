total = 23933.16
rate = 0.3875 / 100.0
month = 664.81

for i in range(36):
    total = total * (1 + rate * 0.7) - month

print(total)
