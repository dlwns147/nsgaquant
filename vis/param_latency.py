import matplotlib.pyplot as plt

fig_path = 'fig/param_latency.png'

params = [0.9, 0.8, 0.7, 0.6, 0.5]

slicegpt_latency = [5.772, 5.265, 4.8636, 4.711, 4.139]
flap_latency = [5.096041055, 4.856390147, 4.384748804, 3.819333236, 3.465446297]

sleb_params = [0.9375, 0.875, 0.84375, 0.8125, 0.75, 0.6875, 0.625, 0.59375, 0.53125, 0.5]
sleb_latency = [4.94, 4.54, 4.48, 4.12, 3.96, 3.65, 3.34, 3.17, 2.86, 2.76]

plt.title('Llama-2-7b')
plt.plot(params, slicegpt_latency, 'o-', label='SliceGPT')
plt.plot(params, flap_latency, 'o-', label='FLAP')
plt.plot(sleb_params, sleb_latency, 'o-', label='Block Pruning')
plt.axhline(5.24, color='grey', linestyle='dashed', alpha=0.5)
# plt.plot([1.0], [5.24], 'o', label='Original')
plt.xlabel('Remained Params (%)')
plt.ylabel('Latency (s)')
plt.legend()
plt.savefig(fig_path, dpi=300)
