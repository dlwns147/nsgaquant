import csv, json
from matplotlib import pyplot as plt
import numpy as np

# awq_7b = '/NAS/SJ/nsgaquant/acl/method/figure_4/awq_random_sample/Llama-2-7b-hf-awq.csv'
# awq_13b = '/NAS/SJ/nsgaquant/acl/method/figure_4/awq_random_sample/Llama-2-13b-hf-awq.csv'
awq_7b = '/NAS/Woo/Automation/autoopt/result/awq_final_search_sample/n_samples_128/Llama-2-7b-hf-awq.csv'
awq_7b_half = '/NAS/Woo/Automation/autoopt/result/awq_final_search_sample/n_samples_128/Llama-2-7b-hf-awq_half.csv'
awq_13b = '/NAS/Woo/Automation/autoopt/result/awq_random_sample/Llama-2-13b-hf-awq.csv'

hqq_7b = '/NAS/Woo/Automation/autoopt/result/hqq_final_sample/replace_Llama-2-7b-hf_wikitext2_ppl.json'
hqq_13b = '/NAS/SJ/nsgaquant/acl/method/figure_4/hqq_ramdon_sample/replace_Llama-2-13b-hf_wikitext2_ppl.json'

awq_7b_data = []
awq_13b_data = []

with open(awq_7b, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        awq_7b_data.append(row)
awq_7b_data = awq_7b_data[1:]
awq_7b_data = [[float(x[1]), float(x[2])] for x in awq_7b_data]

awq_7b_half_data = []
with open(awq_7b_half, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        awq_7b_half_data.append(row)
awq_7b_half_data = awq_7b_half_data[1:]
awq_7b_half_data = [[float(x[1]), float(x[2])] for x in awq_7b_half_data]

# with open(awq_13b, 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         awq_13b_data.append(row)
# awq_13b_data = awq_13b_data[1:]
# awq_13b_data = [[float(x[1]), float(x[2])] for x in awq_13b_data]

with open(hqq_7b, 'r') as f:
    hqq_7b_data = [[float(data['bit']), float(data['ppl'])] for data in json.load(f)['archive']]

# with open(hqq_13b, 'r') as f:
#     hqq_13b_data = [[float(data['bit']), float(data['ppl'])] for data in json.load(f)['archive']]

awq_7b_bits = []; awq_7b_ppl = []; awq_13b_bits = []; awq_13b_ppl = []
hqq_7b_bits = []; hqq_7b_ppl = []; hqq_13b_bits = []; hqq_13b_ppl = []

threshold = 15

for i in range(min(len(awq_7b_data), len(hqq_7b_data))):
    if (awq_7b_data[i][1] < threshold) and (hqq_7b_data[i][1] < threshold):
        awq_7b_bits.append(awq_7b_data[i][0])
        awq_7b_ppl.append(awq_7b_data[i][1])
        hqq_7b_bits.append(hqq_7b_data[i][0])
        hqq_7b_ppl.append(hqq_7b_data[i][1])

for i in range(len(awq_7b_half_data)):
    if (awq_7b_half_data[i][1] < threshold) and (hqq_7b_data[i + 25][1] < threshold):
        awq_7b_bits.append(awq_7b_half_data[i][0])
        awq_7b_ppl.append(awq_7b_half_data[i][1])
        hqq_7b_bits.append(hqq_7b_data[i + 25][0])
        hqq_7b_ppl.append(hqq_7b_data[i + 25][1])

# for i in range(len(awq_13b_data)):
#     if (awq_13b_data[i][1] < threshold) and (hqq_13b_data[i][1] < threshold):
#         awq_13b_bits.append(awq_13b_data[i][0])
#         awq_13b_ppl.append(awq_13b_data[i][1])
#         hqq_13b_bits.append(hqq_13b_data[i][0])
#         hqq_13b_ppl.append(hqq_13b_data[i][1])

# plt.figure(figsize=(10, 6))
font = {'size'   : 15}
plt.rc('font', **font)
plt.rc('axes', axisbelow=True)

plt.figure(figsize=(5, 5))

# ax1.colorbar()
minimum = min(min(hqq_7b_ppl), min(awq_7b_ppl))
maximum = max(max(hqq_7b_ppl), max(awq_7b_ppl))
# ax1.plot([minimum, maximum], [minimum, maximum], 'r-', label='y=x')

x = np.linspace(minimum, maximum, 100)
y = np.emath.logn(1.5, x)
y += minimum - min(y)
plt.plot(x, y, 'r--', label='y=x')

plt.xscale('log')

plt.scatter(hqq_7b_ppl, awq_7b_ppl, c=awq_7b_bits, label='7b')
plt.colorbar()
# plt.scatter(hqq_7b_ppl, awq_7b_ppl, label='7b')

plt.tick_params(axis='both', which='major', labelsize=15)
plt.tick_params(axis='both', which='major', labelsize=15)


# ax2.scatter(hqq_13b_ppl, awq_13b_ppl, c=awq_13b_bits, label='13b')
# ax2.colorbar()
# minimum = min(min(hqq_13b_ppl), min(awq_13b_ppl))
# maximum = max(max(hqq_13b_ppl), max(awq_13b_ppl))
# # ax2.plot([minimum, maximum], [minimum, maximum], 'r-', label='y=x')

# # plot log graph
# x = np.linspace(minimum, maximum, 100)
# # y = np.log2(np.linspace(minimum, maximum, 100))
# y = np.emath.logn(1.5, x)
# y += minimum - min(y)
# ax2.plot(x, y, 'r--', label='y=x')

# plt.xlabel('HQQ')
# plt.ylabel('AWQ')
plt.savefig('source_5.png')
