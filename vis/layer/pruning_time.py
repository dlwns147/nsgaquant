import matplotlib.pyplot as plt
import numpy as np

fig_path = 'fig/layer/pruning_time.png'

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 12))
fig.subplots_adjust(hspace=0.2, wspace=0.15)

finercut_128_7b_param = [0.89621114, 0.80246114, 0.698186528, 0.604436528, 0.500161917]
finercut_512_7b_param = [0.89621114, 0.80246114, 0.698186528, 0.604436528, 0.510686528]
our_layer_7b_128_param = [0.896, 0.8023, 0.6982, 0.6043, 0.5]
our_layer_7b_32_param = [0.896, 0.8023, 0.6982, 0.6043, 0.5]
our_layer_7b_64_param = [0.896, 0.8023, 0.698, 0.6043, 0.5]


finercut_128_7b_w2_ppl = [6.609770298, 8.384570122, 12.12338829, 22.23426819, 49.133358]
finercut_512_7b_w2_ppl = [6.566050053, 8.404876709, 12.25675297, 22.29631424, 63.77688217]
our_layer_7b_128_w2_ppl = [6.419158459, 7.932456017, 12.04807854, 20.24876022, 39.60480881]
our_layer_7b_32_w2_ppl = [6.356456757, 8.074953079, 12.02408791, 20.17902565, 39.02119446]
our_layer_7b_64_w2_ppl = [6.322114468, 8.032236099, 11.50975418, 20.24850845, 40.53092575]

axes[0, 0].plot(finercut_128_7b_param, finercut_128_7b_w2_ppl, alpha=0.5, label='FinerCut* 128')
axes[0, 0].plot(finercut_512_7b_param, finercut_512_7b_w2_ppl, label='FinerCut* 512')
axes[0, 0].plot(our_layer_7b_32_param, our_layer_7b_32_w2_ppl, label='Our Layer 32')
axes[0, 0].plot(our_layer_7b_64_param, our_layer_7b_64_w2_ppl, label='Our Layer 64')
axes[0, 0].plot(our_layer_7b_128_param, our_layer_7b_128_w2_ppl, label='Our Layer 128')
axes[0, 0].set_title("Llama 2 7B Wikitext2 PPL")
axes[0, 0].set_xlabel("Params")
axes[0, 0].set_ylabel("Perplexity")
axes[0, 0].set_yscale('log', base=10)
# axes[0, 0].set_yscale('log', base=e)
axes[0, 0].set_xticks(np.array(range(5, 10)) / 10)
axes[0, 0].grid(c='0.8') 
ins = axes[0, 0].inset_axes([0.5,0.5,0.45,0.45])
# axes[0, 0].legend()

finercut_128_7b_c4_ppl = [8.728996277, 10.96531582, 14.77611828, 23.92844009, 51.02135468]
finercut_512_7b_c4_ppl = []
our_layer_7b_128_c4_ppl = [8.563390732, 10.60588646, 14.35938168, 23.14393425, 42.01548004]
our_layer_7b_32_c4_ppl = [8.619815826, 10.77745724, 14.81438065, 23.46111298, 41.54759216]
our_layer_7b_64_c4_ppl = [8.547148705, 10.73708534, 14.40361977, 23.14447594, 40.1871109]

axes[0, 1].plot(finercut_128_7b_param, finercut_128_7b_c4_ppl, alpha=0.5, label='FinerCut* 128')
axes[0, 1].plot([], finercut_512_7b_c4_ppl, label='FinerCut* 512')
# axes[0, 1].plot(finercut_512_7b_param, finercut_512_7b_c4_ppl, label='FinerCut* 512')
axes[0, 1].plot(our_layer_7b_32_param, our_layer_7b_32_c4_ppl, label='Our Layer 32')
axes[0, 1].plot(our_layer_7b_64_param, our_layer_7b_64_c4_ppl, label='Our Layer 64')
axes[0, 1].plot(our_layer_7b_128_param, our_layer_7b_128_c4_ppl, label='Our Layer 128')
axes[0, 1].set_title("Llama 2 7B C4 PPL")
axes[0, 1].set_xlabel("Params")
axes[0, 1].set_ylabel("Perplexity")
axes[0, 1].set_xticks(np.array(range(5, 10)) / 10)
axes[0, 1].set_yscale('log', base=10)
axes[0, 1].grid(c='0.8') 
# axes[0, 1].legend()

finercut_128_7b_zeroshot = [0.6632, 0.6100, 0.5412, 0.4855, 0.4437]
finercut_512_7b_zeroshot = []
our_layer_7b_128_zeroshot = [0.6521, 0.5970, 0.5355, 0.4958, 0.4313]
our_layer_7b_32_zeroshot = [0.654890107, 0.602273098, 0.540535492, 0.500427064, 0.438724026]
our_layer_7b_64_zeroshot = [0.646052667, 0.591549288, 0.541960829, 0.495135749, 0.426372415]

axes[0, 2].plot(finercut_128_7b_param, finercut_128_7b_zeroshot, alpha=0.5, label='FinerCut* 128')
# axes[0, 2].plot(finercut_512_7b_param, finercut_512_7b_zeroshot, label='FinerCut* 512')
axes[0, 2].plot([], finercut_512_7b_zeroshot, label='FinerCut* 512')
axes[0, 2].plot(our_layer_7b_32_param, our_layer_7b_32_zeroshot, label='Our Layer 32')
axes[0, 2].plot(our_layer_7b_64_param, our_layer_7b_64_zeroshot, label='Our Layer 64')
axes[0, 2].plot(our_layer_7b_128_param, our_layer_7b_128_zeroshot, label='Our Layer 128')
axes[0, 2].set_title("Llama 2 7B Zero-shot")
axes[0, 2].set_xlabel("Params")
axes[0, 2].set_ylabel("Accuracy")
axes[0, 2].set_xticks(np.array(range(5, 10)) / 10)
axes[0, 2].grid(c='0.8') 

finercut_128_13b_param = [0.900826446, 0.800826446, 0.692355372, 0.600619835, 0.49214876]
finercut_512_13b_param = [0.900826446, 0.800826446, 0.700826446, 0.592355372, 0.500619835]
our_layer_13b_128_param = [0.9006, 0.8006, 0.7006, 0.6004, 0.5002]
our_layer_13b_32_param = [0.9006, 0.8008, 0.7006, 0.6006, 0.5004]
our_layer_13b_64_param = [0.9006, 0.8006, 0.7006, 0.6006, 0.5004]

finercut_128_13b_w2_ppl = [5.410739899, 6.434978008, 8.478711128, 12.59064102, 32.46655655]
finercut_512_13b_w2_ppl = [5.410752773, 6.507138252, 9.004765511, 14.90690708, 29.52010345]
our_layer_13b_128_w2_ppl = [5.385161877, 6.329028606, 8.182165146, 11.52717876, 20.820261]
our_layer_13b_32_w2_ppl = [5.405025482, 6.494206905, 8.367053032, 12.85666561, 23.86282921]
our_layer_13b_64_w2_ppl = [5.389243603, 6.339960575, 8.24669075, 12.25156116, 23.33187675]

axes[1, 0].plot(finercut_128_13b_param, finercut_128_13b_w2_ppl, alpha=0.5, label='FinerCut* 128')
axes[1, 0].plot(finercut_512_13b_param, finercut_512_13b_w2_ppl, label='FinerCut* 512')
axes[1, 0].plot(our_layer_13b_32_param, our_layer_13b_32_w2_ppl, label='Our Layer 32')
axes[1, 0].plot(our_layer_13b_64_param, our_layer_13b_64_w2_ppl, label='Our Layer 64')
axes[1, 0].plot(our_layer_13b_128_param, our_layer_13b_128_w2_ppl, label='Our Layer 128')
axes[1, 0].set_title("Llama 2 13B Wikitext2 PPL")
axes[1, 0].set_xlabel("Params")
axes[1, 0].set_ylabel("Perplexity")
axes[1, 0].set_xticks(np.array(range(5, 10)) / 10)
axes[1, 0].set_yscale('log', base=10)
axes[1, 0].grid(c='0.8') 
# axes[1, 0].legend()

finercut_128_13b_c4_ppl = [7.459181786, 8.860198021, 11.39112663, 15.32797527, 35.91783524]
finercut_512_13b_c4_ppl = []
our_layer_13b_128_c4_ppl = [7.4330616, 8.770359993, 10.87189388, 14.5889864, 23.86247826]
our_layer_13b_32_c4_ppl = [7.437764645, 8.894338608, 11.09128666, 15.52100754, 26.62630081]
our_layer_13b_64_c4_ppl = [7.442128181, 8.785021782, 10.90820694, 15.02626991, 30.20301819]

axes[1, 1].plot(finercut_128_13b_param, finercut_128_13b_c4_ppl, alpha=0.5, label='FinerCut* 128')
# axes[1, 1].plot(finercut_512_13b_param, finercut_512_13b_c4_ppl, label='FinerCut* 512')
axes[1, 1].plot([], finercut_512_13b_c4_ppl, label='FinerCut* 512')
axes[1, 1].plot(our_layer_13b_32_param, our_layer_13b_32_c4_ppl, label='Our Layer 32')
axes[1, 1].plot(our_layer_13b_64_param, our_layer_13b_64_c4_ppl, label='Our Layer 64')
axes[1, 1].plot(our_layer_13b_128_param, our_layer_13b_128_c4_ppl, label='Our Layer 128')
axes[1, 1].set_title("Llama 2 13B C4 PPL")
axes[1, 1].set_xlabel("Params")
axes[1, 1].set_ylabel("Perplexity")
axes[1, 1].set_xticks(np.array(range(5, 10)) / 10)
axes[1, 1].set_yscale('log', base=10)
axes[1, 1].grid(c='0.8') 
# axes[1, 1].legend()

finercut_128_13b_zeroshot = [0.7137, 0.6546, 0.6238, 0.5502, 0.4837]
finercut_512_13b_zeroshot = []
our_layer_13b_128_zeroshot = [0.701797034, 0.65809957, 0.626215552, 0.557271158, 0.478276083]
our_layer_13b_32_zeroshot = [0.700450391, 0.653617183, 0.616904005, 0.553539197, 0.481310846]
our_layer_13b_64_zeroshot = [0.706969336, 0.66900912, 0.628666214, 0.55615421, 0.472152817]

axes[1, 2].plot(finercut_128_13b_param, finercut_128_13b_zeroshot, alpha=0.5, label='FinerCut* 128')
# axes[1, 1].plot(finercut_512_13b_param, finercut_512_13b_zeroshot, label='FinerCut* 512')
axes[1, 2].plot([], finercut_512_13b_zeroshot, label='FinerCut* 512')
axes[1, 2].plot(our_layer_13b_32_param, our_layer_13b_32_zeroshot, label='Our Layer 32')
axes[1, 2].plot(our_layer_13b_64_param, our_layer_13b_64_zeroshot, label='Our Layer 64')
axes[1, 2].plot(our_layer_13b_128_param, our_layer_13b_128_zeroshot, label='Our Layer 128')
axes[1, 2].set_title("Llama 2 13B Zero-shot")
axes[1, 2].set_xlabel("Params")
axes[1, 2].set_ylabel("Accuracy")
axes[1, 2].set_xticks(np.array(range(5, 10)) / 10)
axes[1, 2].grid(c='0.8') 

fig.tight_layout() 
fig.subplots_adjust(top=0.90)
# plt.figlegend(lines, labels, loc = 'lower center', ncol=5, labelspacing=0.)
handles, labels = axes[1, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=5, labelspacing=0., fontsize=15)
plt.savefig(fig_path, dpi=300)