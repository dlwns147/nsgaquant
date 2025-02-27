import csv, json
from matplotlib import pyplot as plt
import numpy as np

# awq_7b = '/NAS/SJ/nsgaquant/acl/method/figure_4/awq_random_sample/Llama-2-7b-hf-awq.csv'
# awq_13b = '/NAS/SJ/nsgaquant/acl/method/figure_4/awq_random_sample/Llama-2-13b-hf-awq.csv'
# awq_7b = '/NAS/Woo/Automation/autoopt/result/awq_random_sample/n_samples_128/Llama-2-7b-hf-awq.csv'
# awq_7b_half = '/NAS/Woo/Automation/autoopt/result/awq_random_sample/n_samples_128/Llama-2-7b-hf-awq_half.csv'
# awq_13b = '/NAS/Woo/Automation/autoopt/result/hqq_random_wPrior_sample/replace_Llama-2-13b-hf_wikitext2_ppl.json'
awq_13b_ = '/NAS/Woo/Automation/autoopt/result/awq_random_wPrior_sample/n_samples_128/Llama-2-13b-hf-awq_'

hqq_7b = '/NAS/SJ/nsgaquant/acl/method/figure_4/hqq_ramdon_sample/replace_Llama-2-7b-hf_wikitext2_ppl.json'
hqq_13b = '/NAS/Woo/Automation/autoopt/result/hqq_random_wPrior_sample/replace_Llama-2-13b-hf_wikitext2_ppl.json'

# awq_7b_data = []
awq_13b_data = []

# with open(awq_7b, 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         awq_7b_data.append(row)
# awq_7b_data = awq_7b_data[1:]
# awq_7b_data = [[float(x[1]), float(x[2])] for x in awq_7b_data]

# awq_7b_half_data = []
# with open(awq_7b_half, 'r') as f:
#     reader = csv.reader(f)
#     for row in enumerate(reader):
#         awq_7b_half_data.append(row)
# awq_7b_half_data = awq_7b_half_data[1:]
# awq_7b_half_data = [[float(x[1]), float(x[2])] for x in awq_7b_half_data]

# with open(awq_13b, 'r') as f:
#     reader = csv.reader(f)
#     for row in reader:
#         awq_13b_data.append(row)
# awq_13b_data = awq_13b_data[1:]
# awq_13b_data = [[float(x[1]), float(x[2])] for x in awq_13b_data]

def read_csv(file_path):
    data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    data = data[1:]
    data = [[float(x[1]), float(x[2])] for x in data]
    return data

for i in range(14):
    data = read_csv(awq_13b_ + str(i) + '.csv')
    length = len(data)
    if length <= 40:
        data = data + [[0, 0]] * (40 - length)

    awq_13b_data += data

# with open(hqq_7b, 'r') as f:
#     hqq_7b_data = [[float(data['bit']), float(data['ppl'])] for data in json.load(f)['archive']]

with open(hqq_13b, 'r') as f:
    hqq_13b_data = [[float(data['bit']), float(data['ppl'])] for data in json.load(f)['archive']]

awq_7b_bits = []; awq_7b_ppl = []; awq_13b_bits = []; awq_13b_ppl = []
hqq_7b_bits = []; hqq_7b_ppl = []; hqq_13b_bits = []; hqq_13b_ppl = []

min_threshold = 0
threshold = float('inf')

# for i in range(len(awq_7b_data)):
#     if (awq_7b_data[i][1] < threshold) and (hqq_7b_data[i][1] < threshold):
#         awq_7b_bits.append(awq_7b_data[i][0])
#         awq_7b_ppl.append(awq_7b_data[i][1])
#         hqq_7b_bits.append(hqq_7b_data[i][0])
#         hqq_7b_ppl.append(hqq_7b_data[i][1])

# for i in range(len(awq_7b_half_data)):
#     if (awq_7b_half_data[i][1] < threshold) and (hqq_7b_data[i + 500][1] < threshold):
#         awq_7b_bits.append(awq_7b_half_data[i][0])
#         awq_7b_ppl.append(awq_7b_half_data[i][1])
#         hqq_7b_bits.append(hqq_7b_data[i + 500][0])
#         hqq_7b_ppl.append(hqq_7b_data[i + 500][1])

for i in range(min(len(awq_13b_data), len(hqq_13b_data))):
    if (min_threshold < awq_13b_data[i][1] < threshold) and (min_threshold < hqq_13b_data[i][1] < threshold):
        if awq_13b_data[i][0] == 0:
            continue
        if i == 0: continue
        awq_13b_bits.append(awq_13b_data[i][0])
        awq_13b_ppl.append(awq_13b_data[i][1])
        hqq_13b_bits.append(hqq_13b_data[i][0])
        hqq_13b_ppl.append(hqq_13b_data[i][1])

# plt.figure(figsize=(10, 6))
font = {'size'   : 15}
plt.rc('font', **font)
plt.rc('axes', axisbelow=True)

plt.figure(figsize=(5, 5))

# ax1.colorbar()
minimum = min(min(hqq_13b_ppl), min(awq_13b_ppl))
maximum = max(max(hqq_13b_ppl), max(awq_13b_ppl))
# ax1.plot([minimum, maximum], [minimum, maximum], 'r-', label='y=x')
# plt.plot([minimum, maximum], [minimum, maximum], 'r-', label='y=x')

# x = np.linspace(minimum, maximum, 100)
# y = np.emath.logn(1.5, x)
# y += minimum - min(y)
# plt.plot(x, y, 'r--', label='y=x')

# plt.xscale('log')

# plt.ylim(minimum, 9)
# plt.xlim(minimum, 100)

# plt.plot(hqq_13b_bits, hqq_13b_ppl, 'o', label='HQQ', markersize=5)
# plt.plot(awq_13b_bits, awq_13b_ppl, 'o', label='AWQ', markersize=5)

front_13b_bits = [2.10454545, 2.20309917, 2.25454545, 2.40061983, 2.50433884, 2.60392562, 2.74938017, 2.80206612, 3.00165289, 3.10371901 ,3.20392562, 3.25454545, 3.40123967, 3.50061983, 3.60392562, 3.75475207, 3.80041322, 3.98615702]
hqq_front_13b_ppl = [12.24179459, 9.13484478, 8.461610794, 7.265963078, 6.757240295, 6.330748081, 5.862908363, 5.712350368, 5.407657146, 5.322042942, 5.274187565, 5.243637085, 5.178779602, 5.138917923, 5.096477985, 5.051599503, 5.042751789, 4.997897625]
awq_front_13b_ppl = [7.71299839, 7.209764004, 6.960026741, 6.441867828, 6.211341858, 5.988779068, 5.742465019, 5.634430885, 5.374492168, 5.304297924, 5.24823761, 5.228238106, 5.157777786, 5.128776073, 5.090263367, 5.030594826, 5.024534225, 4.965404987]


plt.scatter(hqq_front_13b_ppl, awq_front_13b_ppl, c=front_13b_bits, label='13b')
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
fig_path = '/NAS/SJ/nsgaquant/acl/method/sample/pareto_front'
plt.savefig(f'{fig_path}.png', dpi=300)
plt.savefig(f'{fig_path}.pdf', dpi=300)
