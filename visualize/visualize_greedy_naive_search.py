import csv
import json
import numpy as np
import matplotlib.pyplot as plt

figure_path = '/NAS/SJ/nsgaquant/fig/greedy_naive_search.png'

model_name='meta-llama/Llama-2-7b-hf'

# greedy_ppl_path = '/NAS/SJ/nsgaquant/csv/greedy_search/Llama-2-7b-hf_ppl_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.csv'
# greedy_ppl_reverse_path = '/NAS/SJ/nsgaquant/csv/greedy_search/Llama-2-7b-hf_reverse_ppl_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_64_sqs_false_sqz_false.csv'

greedy_ppl_path = '/NAS/SJ/nsgaquant/csv/greedy_search/Llama-2-7b-hf_ppl_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_128_sqs_false_sqz_false.csv'
naive_ppl_path = '/NAS/SJ/nsgaquant/csv/naive_search/Llama-2-7b-hf_hqq_ppl_axis_1_lb_4_lgs_128_lqs_false_lqz_false_sb_2_sgs_128_sqs_false_sqz_false.csv'

config='/NAS/SJ/nsgaquant/config/llama.json'
with open(config, 'r') as f:
    config = json.load(f)[model_name]

greedy_outlier_idx = 20
with open(greedy_ppl_path, 'r') as csv_file:
    greedy_ppl_result = list(csv.reader(csv_file))
    greedy_bits = list(map(float, greedy_ppl_result[1]))[:-greedy_outlier_idx]
    greedy_ppl = list(map(float, greedy_ppl_result[2]))[:-greedy_outlier_idx]

with open(naive_ppl_path, 'r') as csv_file:
    naive_ppl_result = list(csv.reader(csv_file))
    naive_bits = list(map(float, naive_ppl_result[1]))[:-greedy_outlier_idx]
    naive_ppl = list(map(float, naive_ppl_result[2]))[:-greedy_outlier_idx]

# plt.scatter(greedy_bits, greedy_ppl, color='g', label='Greedy Search', s=3)
# plt.scatter(naive_bits, naive_ppl, color='b', label='Naive Search', s=3)
plt.plot(greedy_bits, greedy_ppl, color='g', label='Greedy Search')
plt.plot(naive_bits, naive_ppl, color='b', label='Naive Search')
plt.xlabel('Bits', fontsize=14)
plt.ylabel('Perplexity', fontsize=14)
plt.legend(loc="upper right", fontsize=14)
plt.title('Wikitext2 Perplexity', fontsize=15)
plt.show()
plt.savefig(figure_path, dpi=300)