import matplotlib.pyplot as plt
import numpy as np

fig_path = 'fig/layer/latency.png'

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

finercut_7b_w2_ppl = [6.60970974, 8.633400917, 14.69667912, 22.23426819, 49.133358]
finercut_7b_c4_ppl = [8.728960991, 10.95341206, 17.65730476, 23.92844009, 51.02135468]
finercut_7b_zeroshot = [0.663161892, .612785661, 0.548310266, 0.485481327, 0.443681062]
finercut_7b_latency = [4.36374799, 3.910259482, 3.432960401, 2.992824982, 2.57946941]

flap_7b_w2_ppl = [6.084458828, 7.451695919, 10.56966591, 19.85195923, 54.05876923]
flap_7b_c4_ppl = [8.623231888, 11.29648495, 15.55335712, 25.51643372, 52.63236237]
flap_7b_zeroshot = [0.6461, 0.5993, 0.5530, 0.4705, 0.4344]
flap_7b_latency = [5.145940503, 4.85865288, 4.386106094, 3.825875729, 3.447531095]

# our_layer_7b_w2_ppl = [5.877929211, 6.970046043, 9.552556038, 16.44370461, 38.19960022]
# our_layer_7b_c4_ppl = [7.879108429, 9.189109802, 12.09225845, 17.93351936, 38.4600029]
# our_layer_7b_zeroshot = [0.686052469, 0.638005597, 0.567232006, 0.518203724, 0.456754901]
# our_layer_7b_latency = [4.756066766, 4.179966966, 3.663897964, 3.161446759, 2.618085736]

# our_layer_7b_w2_ppl = [6.416173935, 8.071371078, 12.01619053, 20.85782623, 43.51090622]
# our_layer_7b_c4_ppl = [8.605121613, 10.73480988, 14.93680573, 23.45951843, 46.69745636]
# our_layer_7b_zeroshot = [0.64839838, 0.597461185, 0.533808897, 0.495437346, 0.445341458]
# our_layer_7b_latency = [4.56546935, 4.012789191, 3.592867752, 3.174837215, 2.557672821]

our_layer_7b_w2_ppl = [5.997303963, 7.002255917, 9.599042892, 16.74477005, 37.73771667]
our_layer_7b_c4_ppl = [7.966662407, 9.115232468, 12.25316811, 19.92822647, 38.917099]
our_layer_7b_zeroshot = [0.679695772, 0.656400428, 0.569049724, 0.514811483, 0.457738387]
our_layer_7b_latency = [4.627689519, 4.15006549, 3.637264522, 3.112132466, 2.582222241]

sleb_7b_w2_ppl = [6.47, 8.11, 13.82, 29.93, 106.20]
sleb_7b_c4_ppl = [8.71, 10.90, 17.42, 32.06, 85.97]
sleb_7b_zeroshot = [0.6315, 0.5869, 0.5189, 0.4589, 0.4064]
sleb_7b_latency = [4.657030619, 4.217222444, 3.615538829, 3.146555988, 2.662225378]

slicegpt_7b_c4_ppl = [17.3134, 25.8254, 41.2653, 72.8173, 137.7248]
slicegpt_7b_w2_ppl = [5.9564, 6.8606, 8.6269, 12.802, 21.0916]
slicegpt_7b_latency = [5.772, 5.265, 4.8636, 4.711, 4.139]
slicegpt_7b_zeroshot = [0.6236, 0.5664, 0.5168, 0.4505, 0.3960]

axes[0, 0].plot(our_layer_7b_latency, our_layer_7b_c4_ppl, color='blue', label='Our Layer')
axes[0, 0].plot(finercut_7b_latency, finercut_7b_c4_ppl, alpha=0.5, color='purple', label='FinerCut* (Layer)')
axes[0, 0].plot(sleb_7b_latency, sleb_7b_c4_ppl, alpha=0.5, color='brown', label='SLEB (Block)')
axes[0, 0].plot(flap_7b_latency, flap_7b_c4_ppl, alpha=0.5, color='olive', label='FLAP (Channel)')
axes[0, 0].plot(slicegpt_7b_latency, slicegpt_7b_c4_ppl, alpha=0.5, color='brown', label='SliceGPT (Channel)')
axes[0, 0].set_title("Llama 2 7B C4 PPL")
axes[0, 0].set_xlabel("Latency (s)")
axes[0, 0].set_ylabel("Perplexity")
# axes[0, 0].set_yscale('log', base=10)
axes[0, 0].set_ylim([None, 60])
axes[0, 0].grid(c='0.8') 


axes[0, 1].plot(our_layer_7b_latency, our_layer_7b_w2_ppl, color='blue', label='Our Layer')
axes[0, 1].plot(finercut_7b_latency, finercut_7b_w2_ppl, alpha=0.5, color='purple', label='FinerCut*')
axes[0, 1].plot(sleb_7b_latency, sleb_7b_w2_ppl, alpha=0.5, color='brown', label='SLEB')
axes[0, 1].plot(flap_7b_latency, flap_7b_w2_ppl, alpha=0.5, color='olive', label='FLAP (Channel)')
axes[0, 1].plot(slicegpt_7b_latency, slicegpt_7b_w2_ppl, alpha=0.5, color='brown', label='SliceGPT (Channel)')
axes[0, 1].set_title("Llama 2 7B Wikitext2 PPL")
axes[0, 1].set_xlabel("Latency (s)")
axes[0, 1].set_ylabel("Perplexity")
# axes[0, 1].set_yscale('log', base=10)
axes[0, 1].set_ylim([None, 60])
axes[0, 1].grid(c='0.8') 

axes[0, 2].plot(our_layer_7b_latency, our_layer_7b_zeroshot, color='blue', label='Our Layer')
axes[0, 2].plot(finercut_7b_latency, finercut_7b_zeroshot, alpha=0.5, color='purple', label='FinerCut*')
axes[0, 2].plot(sleb_7b_latency, sleb_7b_zeroshot, alpha=0.5, color='brown', label='SLEB')
axes[0, 2].plot(flap_7b_latency, flap_7b_zeroshot, alpha=0.5, color='olive', label='FLAP')
axes[0, 2].plot(slicegpt_7b_latency, slicegpt_7b_zeroshot, alpha=0.5, color='brown', label='SliceGPT (Channel)')
axes[0, 2].set_title("Llama 2 7B Zero-shot task")
axes[0, 2].set_xlabel("Latency (s)")
axes[0, 2].set_ylabel("Mean Accuracy")
axes[0, 2].grid(c='0.8') 

finercut_13b_w2_ppl = [5.410778999, 6.746415615, 10.14544392, 12.59064102, 32.46655655]
finercut_13b_c4_ppl = [7.459159851, 9.318883896, 12.73346806, 15.32797527, 35.91783524]
finercut_13b_zeroshot = [0.71362335, 0.674512839, 0.619253327, 0.550180575, 0.48372558]
finercut_13b_latency = [7.516896643, 6.643129628, 5.794264464, 5.017003249, 4.177446222]

# our_layer_13b_w2 = [5.186352253, 5.524222851, 6.718215942, 9.161161423, 16.90302277]
# our_layer_13b_c4 = [7.095253944, 7.616557121, 8.896360397, 11.69934654, 18.6608963]
# our_layer_13b_zeroshot = [0.701674318, 0.700415676, 0.673493413, 0.611677142, 0.530357119]
# our_layer_13b_latency = [8.33453469, 7.439038731, 6.531702837, 5.588720935, 4.650742928]
# our_layer_13b_w2 = [5.397685051, 6.32451582, 8.229516029, 12.18096066, 25.10446167]
# our_layer_13b_c4 = [7.487156391, 8.847237587, 11.14803696, 15.45329952, 26.41266823]
# our_layer_13b_zeroshot = [0.709464469, 0.666262581, 0.620563087, 0.557366701, 0.488807437]
# our_layer_13b_latency = [7.572344212, 6.698466726, 5.823598005, 5.116008958, 4.235090132]
our_layer_13b_w2_ppl = [5.145305157, 5.538035393, 6.63654089, 9.427079201, 17.38569641]
our_layer_13b_c4_ppl = [7.142875195, 7.599423409, 9.023434639, 11.68260288, 19.00364304]
our_layer_13b_zeroshot = [0.714806186, 0.700866135, 0.654979595, 0.611714982, 0.527561859]
our_layer_13b_latency = [8.295377805, 7.37489574, 6.500791096, 5.568007127, 4.631081003]

sleb_13b_w2_ppl = [5.63, 6.80, 8.64, 12.76, 31.18]
sleb_13b_c4_ppl = [7.8, 9.421603203, 11.61, 16.34, 36.77]
sleb_13b_zeroshot = [0.6672, 0.6299, 0.5881, 0.5374, 0.4583]
sleb_13b_latency = [8.111619976, 7.304090413, 6.372407883, 5.514183403, 4.641428134]

flap_13b_w2_ppl = [5.43, 6.27, 7.70, 11.49, 19.52]
flap_13b_c4_ppl = [7.79, 9.65, 12.26, 17.74, 27.44]
flap_13b_zeroshot = [0.6808, 0.6285, 0.5940, 0.5514, 0.5108]
flap_13b_latency = [8.987817814, 8.359589431, 7.449102472, 6.534335793, 5.538793577]

slicegpt_13b_w2_ppl = [5.2941, 6.0389, 7.4356, 10.6064, 17.5747]
slicegpt_13b_c4_ppl = [15.509, 23.7214, 38.4982, 67.2131, 121.7989]
slicegpt_13b_zeroshot = [0.6843, 0.6238, 0.5467, 0.4727, 0.4140]
slicegpt_13b_latency = [9.2124, 8.7407, 7.9494, 7.3503, 7.1742]

axes[1, 0].plot(our_layer_13b_latency, our_layer_13b_c4_ppl, color='blue', label='Our Layer')
axes[1, 0].plot(finercut_13b_latency, finercut_13b_c4_ppl, alpha=0.5, color='purple', label='FinerCut*')
axes[1, 0].plot(sleb_13b_latency, sleb_13b_c4_ppl, alpha=0.5, color='brown', label='SLEB')
axes[1, 0].plot(flap_13b_latency, flap_13b_c4_ppl, alpha=0.5, color='olive', label='FLAP')
axes[1, 0].plot(slicegpt_13b_latency, slicegpt_13b_c4_ppl, alpha=0.5, color='brown', label='SliceGPT (Channel)')
axes[1, 0].set_title("Llama 2 13b C4 PPL")
axes[1, 0].set_xlabel("Latency (s)")
axes[1, 0].set_ylabel("Perplexity")
axes[1, 0].set_ylim([None, 40])
# axes[1, 0].set_yscale('log', base=10)
axes[1, 0].grid(c='0.8') 


axes[1, 1].plot(our_layer_13b_latency, our_layer_13b_w2_ppl, color='blue', label='Our Layer')
axes[1, 1].plot(finercut_13b_latency, finercut_13b_w2_ppl, alpha=0.5, color='purple', label='FinerCut*')
axes[1, 1].plot(sleb_13b_latency, sleb_13b_w2_ppl, alpha=0.5, color='brown', label='SLEB')
axes[1, 1].plot(flap_13b_latency, flap_13b_w2_ppl, alpha=0.5, color='olive', label='FLAP')
axes[1, 1].plot(slicegpt_13b_latency, slicegpt_13b_w2_ppl, alpha=0.5, color='brown', label='SliceGPT (Channel)')
axes[1, 1].set_title("Llama 2 13B Wikitext2 PPL")
axes[1, 1].set_xlabel("Latency (s)")
axes[1, 1].set_ylabel("Perplexity")
# axes[1, 1].set_yscale('log', base=10)
axes[1, 1].grid(c='0.8') 

axes[1, 2].plot(our_layer_13b_latency, our_layer_13b_zeroshot, color='blue', label='Our Layer')
axes[1, 2].plot(finercut_13b_latency, finercut_13b_zeroshot, alpha=0.5, color='purple', label='FinerCut*')
axes[1, 2].plot(sleb_13b_latency, sleb_13b_zeroshot, alpha=0.5, color='brown', label='SLEB')
axes[1, 2].plot(flap_13b_latency, flap_13b_zeroshot, alpha=0.5, color='olive', label='FLAP')
axes[1, 2].plot(slicegpt_13b_latency, slicegpt_13b_zeroshot, alpha=0.5, color='brown', label='SliceGPT (Channel)')
axes[1, 2].set_title("Llama 2 13B Zero-shot task")
axes[1, 2].set_xlabel("Latency (s)")
axes[1, 2].set_ylabel("Mean Accuracy")
axes[1, 2].grid(c='0.8') 

fig.tight_layout() 
fig.subplots_adjust(top=0.9)
# plt.figlegend(lines, labels, loc = 'lower center', ncol=5, labelspacing=0.)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(labels), labelspacing=0., fontsize=15)
plt.savefig(fig_path, dpi=300)