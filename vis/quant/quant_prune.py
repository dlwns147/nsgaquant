import matplotlib.pyplot as plt

fig_path='fig/quant/quant_prune.png'

quant_hqq_7b_w2_ppl = [12.35548496, 8.825408936, 7.231790066, 6.384272575, 6.095602036, 5.920915604, 5.808477879]
quant_hqq_7b_c4_ppl = [16.35224915, 12.03056145, 9.906867027, 8.648254395, 8.202033043, 7.906612873, 7.719115257]
quant_hqq_7b_acc = [0.55073265, 0.601491773, 0.627732695, 0.648273017, 0.666196182, 0.667744791, 0.675904417]
quant_hqq_7b_bits = [2.249313117, 2.504844863, 2.749645806, 3.003969499, 3.25280825, 3.500985417, 3.748186023]

quant_prune_hqq_7b_w2_ppl = [15.28255463, 10.97931767, 8.601859093, 7.178581238, 6.402769566, 6.110738754, 5.920007229, 5.813964367]
quant_prune_hqq_7b_c4_ppl = [19.29504585, 14.37425327, 11.5943203, 9.742159843, 8.647332191, 8.195744514, 7.934197426, 7.720817566]
quant_prune_hqq_7b_acc = [0.507203384, 0.556525661, 0.60140474, 0.637681886, 0.655881502, 0.667877683, 0.669455122, 0.680571698]
quant_prune_hqq_7b_bits = [1.99838083, 2.25323834, 2.49967617, 2.752266839, 3.00080959, 3.25242876, 3.50404793, 3.75420984]

quant_hqq_13b_w2_ppl = [8.30193615, 6.696972847, 5.883749485, 5.449792385, 5.233938694, 5.135648727, 5.047449589]
quant_hqq_13b_c4_ppl = [11.37708378, 9.290312767, 8.167218208, 7.53046751, 7.222092628, 7.082673073, 6.967734814]
quant_hqq_13b_acc = [0.617395165, 0.659029422, 0.678651527, 0.689248983, 0.69588637, 0.704057488, 0.708629525]
quant_hqq_13b_bits = [2.253367769, 2.503933368, 2.750076188, 3.004018595, 3.252374742, 3.503573089, 3.752493543]

quant_prune_hqq_13b_w2_ppl = [9.434177399, 7.693449497, 6.635149956, 5.866407871, 5.459187984, 5.258904457, 5.133044243, 5.049158096]
quant_prune_hqq_13b_c4_ppl = [12.71516323, 10.5737009, 9.087550163, 8.115990639, 7.52857399, 7.26716423, 7.097723007, 6.969366074]
quant_prune_hqq_13b_acc = [0.594325206, 0.633129268, 0.670933528, 0.680546663, 0.691661709, 0.699375713, 0.704672332, 0.708850504]
quant_prune_hqq_13b_bits = [2.00289256, 2.253719008, 2.5035124, 2.75413223, 3.00309917, 3.254958678, 3.503719008, 3.753099174]

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10)) 
fig.subplots_adjust(hspace=0.5, wspace=0.1)

# axes[0, 0].plot(pbllm_7b_bits, pbllm_7b_w2_ppl, label='pb-llm')
axes[0, 0].plot(quant_hqq_7b_bits, quant_hqq_7b_w2_ppl, label='quant')
axes[0, 0].plot(quant_prune_hqq_7b_bits, quant_prune_hqq_7b_w2_ppl, label='quant+prune')
axes[0, 0].set_title('Llama 2 7B Wikitext')
axes[0, 0].set_xlabel('Bits')
axes[0, 0].set_ylabel('Perplexity')
axes[0, 0].grid(c='0.8')
axes[0, 0].legend(loc='upper right')

# axes[0, 1].plot(pbllm_7b_bits, pbllm_7b_w2_ppl, label='pb-llm')
axes[0, 1].plot(quant_hqq_7b_bits, quant_hqq_7b_c4_ppl, label='quant')
axes[0, 1].plot(quant_prune_hqq_7b_bits, quant_prune_hqq_7b_c4_ppl, label='quant+prune')
axes[0, 1].set_title('Llama 2 7B C4')
axes[0, 1].set_xlabel('Bits')
axes[0, 1].set_ylabel('Perplexity')
axes[0, 1].grid(c='0.8')
axes[0, 1].legend(loc='upper right')

axes[0, 2].plot(quant_hqq_7b_bits, quant_hqq_7b_acc, label='quant')
axes[0, 2].plot(quant_prune_hqq_7b_bits, quant_prune_hqq_7b_acc, label='quant+prune')
axes[0, 2].set_title('Llama 2 7B 0-shot Avg')
axes[0, 2].set_xlabel('Bits')
axes[0, 2].set_ylabel('Acc.')
axes[0, 2].grid(c='0.8')
axes[0, 2].legend(loc='lower right')

axes[1, 0].plot(quant_hqq_13b_bits, quant_hqq_13b_w2_ppl, label='quant')
axes[1, 0].plot(quant_prune_hqq_13b_bits, quant_prune_hqq_13b_w2_ppl, label='quant+prune')
axes[1, 0].set_title('Llama 2 13B Wikitext')
axes[1, 0].set_xlabel('Bits')
axes[1, 0].set_ylabel('Perplexity')
axes[1, 0].grid(c='0.8')
axes[1, 0].legend(loc='upper right')

# axes[0, 1].plot(pbllm_13b_bits, pbllm_13b_w2_ppl, label='pb-llm')
axes[1, 1].plot(quant_hqq_13b_bits, quant_hqq_13b_c4_ppl, label='quant')
axes[1, 1].plot(quant_prune_hqq_13b_bits, quant_prune_hqq_13b_c4_ppl, label='quant+prune')
axes[1, 1].set_title('Llama 2 13B C4')
axes[1, 1].set_xlabel('Bits')
axes[1, 1].set_ylabel('Perplexity')
axes[1, 1].grid(c='0.8')
axes[1, 1].legend(loc='upper right')

axes[1, 2].plot(quant_hqq_13b_bits, quant_hqq_13b_acc, label='quant')
axes[1, 2].plot(quant_prune_hqq_13b_bits, quant_prune_hqq_13b_acc, label='quant+prune')
axes[1, 2].set_title('Llama 2 13B 0-shot Avg')
axes[1, 2].set_xlabel('Bits')
axes[1, 2].set_ylabel('Acc.')
axes[1, 2].grid(c='0.8')
axes[1, 2].legend(loc='lower right')


fig.tight_layout() 
plt.savefig(fig_path, dpi=300)



