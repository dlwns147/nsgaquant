import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from datetime import datetime

now = datetime.now() # current date and time
date_time = now.strftime("%Y%m%d%H%M%S")

# sleb_figure = f'fig/layer/{date_time}_total.png'
# sleb_figure = f'fig/layer/total.png'
sleb_figure = f'fig/layer/total.png'
# figure_path = f'fig/layer/compare'



# finercut_7b_w2_ppl = "6.609770298	8.384570122	12.12338829	22.23426819	49.133358"
# finercut_7b_w2_ppl = list(map(float, finercut_7b_w2_ppl.split("\t")))
# finercut_7b_c4_ppl = "8.728996277	10.96531582	14.77611828	23.92844009	51.02135468"
# finercut_7b_c4_ppl = list(map(float, finercut_7b_c4_ppl.split("\t")))
# finercut_7b_param = "0.89621114	0.80246114	0.698186528	0.604436528	0.500161917"
# finercut_7b_param = list(map(float, finercut_7b_param.split("\t")))
# finercut_7b_zeroshot = [0.6632, 0.6100, 0.5412, 0.4855, 0.4437]
# finercut_7b_latency = "4.447615644	4.155571699	3.954386498	3.72502402	3.506142619	3.301202521	3.025111539	2.855957752	2.596383587"
# finercut_7b_latency = list(map(float, finercut_7b_latency.split("\t")))

# our_layer_7b_w2_ppl = "6.419158459	7.932456017	12.04807854	20.24876022	39.60480881"
# our_layer_7b_w2_ppl = list(map(float, our_layer_7b_w2_ppl.split("\t")))
# our_layer_7b_c4_ppl = "8.563390732	10.60588646	14.35938168	23.14393425	42.01548004"
# our_layer_7b_c4_ppl = list(map(float, our_layer_7b_c4_ppl.split("\t")))
# our_layer_7b_zeroshot = [0.6521, 0.5970, 0.5355, 0.4958, 0.4313]
# our_layer_7b_param = "0.896	0.8023	0.6982	0.6043	0.5"
# our_layer_7b_param = list(map(float, our_layer_7b_param.split("\t")))
# our_layer_7b_latency = "4.65	4.25	4.06	3.71	3.49	3.32	3.12	2.92	2.7"
# our_layer_7b_latency = list(map(float, our_layer_7b_latency.split("\t")))


# our_layer_7b_w2_ppl = [6.383822441, 7.932436943, 11.48663998, 20.63082123, 42.38755417]
# our_layer_7b_c4_ppl = [8.589793205, 10.60577011, 14.7732439, 23.2828598, 48.81367874]
# our_layer_7b_zeroshot = [0.650455237, 0.59805192, 0.5417561, 0.496703311, 0.449992264]
# our_layer_7b_param = [0.896, 0.8023, 0.698, 0.6043, 0.5002]

finercut_7b_w2_ppl = [6.60970974, 8.633400917, 14.69667912, 22.23426819, 49.133358]
finercut_7b_c4_ppl = [8.728960991, 10.95341206, 17.65730476, 23.92844009, 51.02135468]
finercut_7b_zeroshot = [0.663161892, .612785661, 0.548310266, 0.485481327, 0.443681062]
finercut_7b_param = [0.89621114, 0.80246114, 0.70871114, 0.604436528, 0.500161917]

our_layer_7b_w2_ppl = [6.416173935, 8.071371078, 12.01619053, 20.85782623, 43.51090622]
our_layer_7b_c4_ppl = [8.605121613, 10.73480988, 14.93680573, 23.45951843, 46.69745636]
our_layer_7b_zeroshot = [0.64839838, 0.597461185, 0.533808897, 0.495437346, 0.445341458]
our_layer_7b_param = [0.896049223, 0.802299223, 0.698024611, 0.604274611, 0.500161917]

sleb_7b_w2_ppl = "6.47	8.11	13.82	29.93	106.20"
sleb_7b_w2_ppl = list(map(float, sleb_7b_w2_ppl.split("\t")))
sleb_7b_c4_ppl = "8.71	10.90	17.42	32.06	85.97"
sleb_7b_c4_ppl = list(map(float, sleb_7b_c4_ppl.split("\t")))
sleb_7b_zeroshot = [0.6632, 0.6100, 0.5412, 0.4855, 0.4437]
sleb_7b_param = "0.90625	0.8125	0.6875	0.59375	0.5"
sleb_7b_param = list(map(float, sleb_7b_param.split("\t")))
sleb_7b_zeroshot = [0.6315, 0.5869, 0.5189, 0.4589, 0.4064]

our_block_7b_w2_ppl = '6.346566677	8.096574783	13.0532198	26.14618111	66.43428802'
our_block_7b_w2_ppl = list(map(float, our_block_7b_w2_ppl.split("\t")))
our_block_7b_c4_ppl = '8.515769005	10.61856747	15.49253559	28.15706062	65.44973755'
our_block_7b_c4_ppl = list(map(float, our_block_7b_c4_ppl.split("\t")))
our_block_7b_zeroshot = [0.6465, 0.5872, 0.5206, 0.4758, 0.4376]
our_block_7b_param = "0.90625	0.8125	0.6875	0.59375	0.5"
our_block_7b_param = list(map(float, our_block_7b_param.split("\t")))

# our_block_7b_w2_ppl = [6.383822441, 7.932436943, 11.48663998, 20.63082123, 42.38755417]
# our_block_7b_c4_ppl = [8.589793205, 10.60577011, 14.7732439, 23.2828598, 48.81367874]
# our_block_7b_zeroshot = [0.650455237, 0.59805192, 0.5417561, 0.496703311, 0.449992264]
# our_block_7b_param = [0.896, 0.8023, 0.698, 0.6043, 0.5002]

slicegpt_7b_param = [0.9, 0.8, 0.7, 0.6, 0.5]
slicegpt_7b_c4_ppl = [17.3134, 25.8254, 41.2653, 72.8173, 137.7248]
slicegpt_7b_w2_ppl = [5.9564, 6.8606, 8.6269, 12.802, 21.0916]
slicegpt_7b_latency = [5.772, 5.265, 4.8636, 4.711, 4.139]
slicegpt_7b_zeroshot = [0.6236, 0.5664, 0.5168, 0.4505, 0.3960]

flap_7b_param = [0.9, 0.8, 0.7, 0.6, 0.5]
flap_7b_c4_ppl = [8.623231888, 11.29648495, 15.55335712, 25.51643372, 52.63236237]
flap_7b_w2_ppl = [6.084458828, 7.451695919, 10.56966591, 19.85195923, 54.05876923]
# flap_7b_latency = [5.087628107, 4.818382587, 4.320798185, 3.788519752, 3.425058297]
flap_7b_zeroshot = [0.6461, 0.5993, 0.5530, 0.4705, 0.4344]
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 20))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

axes[0, 0].plot(finercut_7b_param, finercut_7b_c4_ppl, color='purple', label='FinerCut* (Layer)')
axes[0, 0].plot(slicegpt_7b_param, slicegpt_7b_c4_ppl, color='orange', label='SliceGPT (Channel)')
axes[0, 0].plot(flap_7b_param, flap_7b_c4_ppl, color='olive', label='FLAP (Channel)')
axes[0, 0].plot(sleb_7b_param, sleb_7b_c4_ppl, color='brown', label='SLEB (Block)')
axes[0, 0].plot(our_block_7b_param, our_block_7b_c4_ppl, color='teal', label='Our Block')
axes[0, 0].plot(our_layer_7b_param, our_layer_7b_c4_ppl, color='blue', label='Our Layer')
axes[0, 0].set_title(f'Llama-2-7b')
axes[0, 0].set_xlabel('Params')
axes[0, 0].set_ylabel('C4 Perplexity (PPL)')
axes[0, 0].set_xticks(np.array(range(5, 10)) / 10)
axes[0, 0].set_yticks(np.array(range(0, 100, 20)))
# axes[0, 0].set_yscale('log', base=10)
axes[0, 0].set_ylim([5, 80])
axes[0, 0].grid(c='0.8')
# axes[0, 0].legend(loc="upper right")


axes[0, 1].plot(finercut_7b_param, finercut_7b_w2_ppl, color='purple', label='FinerCut* (Layer)')
# axes[0, 1].plot(slicegpt_7b_param, slicegpt_7b_w2_ppl, color='orange', label='SliceGPT (Channel)')
# axes[0, 1].plot(flap_7b_param, flap_7b_w2_ppl, color='olive', label='FLAP (Channel)')
axes[0, 1].plot([], [], color='orange', label='SliceGPT (Channel)')
axes[0, 1].plot([], [], color='olive', label='FLAP (Channel)')
axes[0, 1].plot(sleb_7b_param, sleb_7b_w2_ppl, color='brown', label='SLEB (Block)')
axes[0, 1].plot(our_block_7b_param, our_block_7b_w2_ppl, color='teal', label='Our Block')
axes[0, 1].plot(our_layer_7b_param, our_layer_7b_w2_ppl, color='blue', label='Our Layer')
axes[0, 1].set_title(f'Llama-2-7b')
axes[0, 1].set_xlabel('Params')
axes[0, 1].set_ylabel('Wikitext2 Perplexity (PPL)')
axes[0, 1].set_xticks(np.array(range(5, 10)) / 10)
axes[0, 1].set_yticks(np.array(range(0, 100, 20)))
# axes[0, 1].set_yscale('log', base=10)
axes[0, 1].set_ylim([5, 80])
axes[0, 1].grid(c='0.8')
# axes[0, 1].legend(loc="upper right")


axes[0, 2].plot(finercut_7b_param, finercut_7b_zeroshot, color='purple', label='FinerCut* (Layer)')
axes[0, 2].plot(our_layer_7b_param, our_layer_7b_zeroshot, color='blue', label='Our Layer')
axes[0, 2].plot(slicegpt_7b_param, slicegpt_7b_zeroshot, color='orange', label='SliceGPT (Channel)')
axes[0, 2].plot(flap_7b_param, flap_7b_zeroshot, color='olive', label='FLAP (Channel)')
axes[0, 2].plot(our_block_7b_param, our_block_7b_zeroshot, color='teal', label='Our Block')
axes[0, 2].plot(sleb_7b_param, sleb_7b_zeroshot, color='brown', label='SLEB (Block)')
axes[0, 2].set_title(f'Llama-2-7b')
axes[0, 2].set_xlabel('Params')
axes[0, 2].set_ylabel('Zeroshot task Mean Acc.')
# axes[0, 2].set_yticks(np.array(range(0, 100, 5)))
axes[0, 2].set_xticks(np.array(range(5, 10)) / 10)
axes[0, 2].grid(c='0.8')
# axes[0, 2].set_ylim([None, 50])
# axes[0, 2].legend(loc="upper left")


# finercut_13b_w2_ppl = "5.410739899	6.434978008	8.478711128	12.59064102	32.46655655"
# finercut_13b_w2_ppl = list(map(float, finercut_13b_w2_ppl.split("\t")))
# finercut_13b_c4_ppl = "7.459181786	8.860198021	11.39112663	15.32797527	35.91783524"
# finercut_13b_c4_ppl = list(map(float, finercut_13b_c4_ppl.split("\t")))
# finercut_13b_param = "0.900826446	0.800826446	0.692355372	0.600619835	0.49214876"
# finercut_13b_param = list(map(float, finercut_13b_param.split("\t")))
# finercut_13b_zeroshot = [0.7137, 0.6546, 0.6238, 0.5502, 0.4837]
# finercut_13b_latency = "7.504453059	7.056995951	6.614486172	6.147576247	5.750985445	5.444466345	5.022227337	4.565828106	4.154034714"
# finercut_13b_latency = list(map(float, finercut_13b_latency.split("\t")))

# our_layer_13b_w2_ppl = "5.385161877	6.329028606	8.182165146	11.52717876	20.820261"
# our_layer_13b_w2_ppl = list(map(float, our_layer_13b_w2_ppl.split("\t")))
# our_layer_13b_c4_ppl = "7.4330616	8.770359993	10.87189388	14.5889864	23.86247826"
# our_layer_13b_c4_ppl = list(map(float, our_layer_13b_c4_ppl.split("\t")))
# our_layer_13b_param = "0.9006	0.8006	0.7006	0.6004	0.5002"
# our_layer_13b_param = list(map(float, our_layer_13b_param.split("\t")))
# our_layer_13b_zeroshot = [0.7018, 0.6581, 0.6262, 0.5573, 0.4783]
# our_layer_13b_latency = "7.63	7.27	6.7	6.42	5.88	5.56	5.12	4.79	4.43"
# our_layer_13b_latency = list(map(float, our_layer_13b_latency.split("\t")))

# our_layer_13b_w2_ppl = [5.397685051, 6.32451582, 8.229516029, 12.18096066, 25.10446167]
# our_layer_13b_c4_ppl = [7.487156391, 8.847237587, 11.14803696, 15.45329952, 26.41266823]
# our_layer_13b_zeroshot = [0.709464469, 0.666262581, 0.620563087, 0.557366701, 0.488807437]
# our_layer_13b_param = [0.9006, 0.8006, 0.7006, 0.6004, 0.5004]

finercut_13b_w2_ppl = [5.410778999, 6.746415615, 10.14544392, 12.59064102, 32.46655655]
finercut_13b_c4_ppl = [7.459159851, 9.318883896, 12.73346806, 15.32797527, 35.91783524]
finercut_13b_zeroshot = [0.71362335, 0.674512839, 0.619253327, 0.550180575, 0.48372558]
finercut_13b_param = [0.900826446, 0.800826446, 0.692355372, 0.600619835, 0.49214876]

our_layer_13b_w2_ppl = [5.393065929, 6.444874763, 8.248177528, 12.25156116, 24.1680088]
our_layer_13b_c4_ppl = [7.435959816, 8.935520172, 11.40456963, 15.02626991, 25.42998505]
our_layer_13b_zeroshot = [0.702718204, 0.646050357, 0.591392236, 0.55615421, 0.479214489]
our_layer_13b_param = [0.900619835, 0.800619835, 0.700413223, 0.600619835, 0.500413223]

slicegpt_13b_param = [0.9, 0.8, 0.7, 0.6, 0.5]
slicegpt_13b_w2_ppl = [5.2941, 6.0389, 7.4356, 10.6064, 17.5747]
slicegpt_13b_c4_ppl = [15.509, 23.7214, 38.4982, 67.2131, 121.7989]
slicegpt_13b_latency = [9.2124, 8.7407, 7.9494, 7.3503, 7.1742]
slicegpt_13b_zeroshot = [0.6843, 0.6238, 0.5467, 0.4727, 0.4140]

flap_13b_param = [0.9, 0.8, 0.7, 0.6, 0.5]
flap_13b_w2_ppl = [5.43, 6.27, 7.70, 11.49, 19.52]
flap_13b_c4_ppl = [7.79, 9.65, 12.26, 17.74, 27.44]
# flap_13b_ latency = [8.789877715, 8.15873653, 7.382896387, 6.396090912, 5.435466782]
flap_13b_zeroshot = [0.6808, 0.6285, 0.5940, 0.5514, 0.5108]

sleb_13b_w2_ppl = "5.63	6.80	8.64	12.76	31.18"
sleb_13b_w2_ppl = list(map(float, sleb_13b_w2_ppl.split("\t")))
sleb_13b_c4_ppl = "7.8	9.42	11.61	16.34	36.77"
sleb_13b_c4_ppl = list(map(float, sleb_13b_c4_ppl.split("\t")))
sleb_13b_param = "0.9	0.8	0.7	0.6	0.5"
sleb_13b_param = list(map(float, sleb_13b_param.split("\t")))
sleb_13b_zeroshot = [0.6672, 0.6299, 0.5881, 0.5374, 0.4583]

our_block_13b_w2_ppl = "5.608878613	6.881039619	8.682220459	12.66842079	28.48030663"
our_block_13b_w2_ppl = list(map(float, our_block_13b_w2_ppl.split("\t")))
our_block_13b_c4_ppl = "7.815994263	9.416050911	11.65200424	16.25538826	30.09889221"
our_block_13b_c4_ppl = list(map(float, our_block_13b_c4_ppl.split("\t")))
our_block_13b_param = "0.9	0.8	0.7	0.6	0.5"
our_block_13b_param = list(map(float, our_block_13b_param.split("\t")))
our_block_13b_zeroshot = [0.6882, 0.6318, 0.5931, 0.5423, 0.4675]

axes[1, 0].plot(finercut_13b_param, finercut_13b_c4_ppl, color='purple', label='FinerCut* (Layer)')
axes[1, 0].plot(slicegpt_13b_param, slicegpt_13b_c4_ppl, color='orange', label='SliceGPT (Channel)')
axes[1, 0].plot(flap_13b_param, flap_13b_c4_ppl, color='olive', label='FLAP (Channel)')
axes[1, 0].plot(sleb_13b_param, sleb_13b_c4_ppl, color='brown', label='SLEB (Block)')
axes[1, 0].plot(our_block_13b_param, our_block_13b_c4_ppl, color='teal', label='Our Block')
axes[1, 0].plot(our_layer_13b_param, our_layer_13b_c4_ppl, color='blue', label='Our Layer')
axes[1, 0].set_title(f'Llama-2-13b')
axes[1, 0].set_xlabel('Params')
axes[1, 0].set_ylabel('C4 Perplexity (PPL)')
axes[1, 0].set_xticks(np.array(range(5, 10)) / 10)
axes[1, 0].set_yticks(np.array(range(0, 50, 10)))
# axes[1, 0].set_yscale('log', base=10)
axes[1, 0].set_ylim([5, 40])
axes[1, 0].grid(c='0.8')
# axes[1, 0].legend(loc="upper right")

axes[1, 1].plot(finercut_13b_param, finercut_13b_w2_ppl, color='purple', label='FinerCut* (Layer)')
axes[1, 1].plot(sleb_13b_param, sleb_13b_w2_ppl, color='brown', label='SLEB (Block)')
# axes[1, 1].plot(slicegpt_13b_param, slicegpt_13b_w2_ppl, color='orange', label='SliceGPT (Channel)')
# axes[1, 1].plot(flap_13b_param, flap_13b_w2_ppl, color='olive', label='FLAP (Channel)')
axes[1, 1].plot([], [], color='orange', label='SliceGPT (Channel)')
axes[1, 1].plot([], [], color='olive', label='FLAP (Channel)')
axes[1, 1].plot(our_block_13b_param, our_block_13b_w2_ppl, color='teal', label='Our Block')
axes[1, 1].plot(our_layer_13b_param, our_layer_13b_w2_ppl, color='blue', label='Our Layer')
axes[1, 1].set_title(f'Llama-2-13b')
axes[1, 1].set_xlabel('Params')
axes[1, 1].set_ylabel('Wikitext2 Perplexity (PPL)')
axes[1, 1].set_xticks(np.array(range(5, 10)) / 10)
axes[1, 1].set_yticks(np.array(range(0, 40, 10)))
# axes[1, 1].set_yscale('log', base=10)
axes[1, 1].set_ylim([5, 35])
axes[1, 1].grid(c='0.8')
# axes[1, 1].legend(loc="upper right")


axes[1, 2].plot(finercut_13b_param, finercut_13b_zeroshot, color='purple', label='FinerCut* (Layer)')
axes[1, 2].plot(slicegpt_13b_param, slicegpt_13b_zeroshot, color='orange', label='SliceGPT (Channel)')
axes[1, 2].plot(flap_13b_param, flap_13b_zeroshot, color='olive', label='FLAP (Channel)')
axes[1, 2].plot(sleb_13b_param, sleb_13b_zeroshot, color='brown', label='SLEB (Block)')
axes[1, 2].plot(our_block_13b_param, our_block_13b_zeroshot, color='teal', label='Our Block')
axes[1, 2].plot(our_layer_13b_param, our_layer_13b_zeroshot, color='blue', label='Our Layer')
axes[1, 2].set_title(f'Llama-2-13b')
axes[1, 2].set_xlabel('Params')
axes[1, 2].set_ylabel('Zeroshot task Mean Acc.')
axes[1, 2].set_xticks(np.array(range(5, 10)) / 10)
axes[1, 2].grid(c='0.8')
# axes[1, 2].set_ylim([5, 35])
# axes[1, 2].legend(loc="upper left")


finercut_70b_w2_ppl = "4.13	5.11	6.34	8.85	14.61"
finercut_70b_w2_ppl = list(map(float, finercut_70b_w2_ppl.split("\t")))
finercut_70b_c4_ppl = "6.52	7.49	8.84	11.62	21.29"
finercut_70b_c4_ppl = list(map(float, finercut_70b_c4_ppl.split("\t")))
finercut_70b_param = "0.896323529	0.800735294	0.705147059	0.601470588	0.5"
finercut_70b_param = list(map(float, finercut_70b_param.split("\t")))
finercut_70b_zeroshot = [0.7377, 0.6978, 0.6622, 0.5955, 0.5369]
# finercut_70b_latency = "7.504453059	7.056995951	6.614486172	6.147576247	5.750985445	5.444466345	5.022227337	4.565828106	4.154034714"
# finercut_70b_latency = list(map(float, finercut_70b_latency.split("\t")))

our_layer_70b_w2_ppl = "3.89	4.847938061	6.069574356	8.248405457	11.61444473"
our_layer_70b_w2_ppl = list(map(float, our_layer_70b_w2_ppl.split("\t")))
our_layer_70b_c4_ppl = "6.25	7.308520317	8.610116959	11.00120449	14.39074039"
our_layer_70b_c4_ppl = list(map(float, our_layer_70b_c4_ppl.split("\t")))
our_layer_70b_param = "0.9044	0.8037	0.7037	0.6037	0.5022"
our_layer_70b_param = list(map(float, our_layer_70b_param.split("\t")))
our_layer_70b_zeroshot = [0.7424, 0.7121, 0.6759, 0.6118, 0.5547]
# our_layer_70b_latency = "7.63	7.27	6.7	6.42	5.88	5.56	5.12	4.79	4.43"
# our_layer_70b_latency = list(map(float, our_layer_70b_latency.split("\t")))

slicegpt_70b_param = [0.9, 0.8, 0.7, 0.6, 0.5]
slicegpt_70b_w2_ppl = [3.7762, 4.4565, 5.4185, 7.091, 10.753]
slicegpt_70b_c4_ppl = [10.0544, 15.7577, 25.8468, 46.9589, 89.7559]
slicegpt_70b_latency = [9.2124, 8.7407, 7.9494, 7.3503, 7.1742]
slicegpt_70b_zeroshot = [0.7545, 0.7261, 0.6616, 0.5709, 0.4742]

sleb_70b_w2_ppl = "3.98	4.88	5.93	7.57	11.58"
sleb_70b_w2_ppl = list(map(float, sleb_70b_w2_ppl.split("\t")))
sleb_70b_c4_ppl = "6.32	7.306227684	8.64	12.16	19.68"
sleb_70b_c4_ppl = list(map(float, sleb_70b_c4_ppl.split("\t")))
sleb_70b_param = "0.9	0.8	0.7	0.6	0.5"
sleb_70b_param = list(map(float, sleb_70b_param.split("\t")))
sleb_70b_zeroshot = [0.7314, 0.7083, 0.6737, 0.6239, 0.5651]

our_block_70b_w2_ppl = "3.940636635	4.842317104	5.900648117	7.503790379	10.99170876"
our_block_70b_w2_ppl = list(map(float, our_block_70b_w2_ppl.split("\t")))
our_block_70b_c4_ppl = "6.282603264	7.279605865	8.623544693	10.67622852	17.55974579"
our_block_70b_c4_ppl = list(map(float, our_block_70b_c4_ppl.split("\t")))
our_block_70b_param = "0.9	0.8	0.7	0.6	0.5"
our_block_70b_param = list(map(float, our_block_70b_param.split("\t")))
our_block_70b_zeroshot = [0.7457, 0.7030, 0.6688, 0.6334, 0.5666]

axes[2, 0].plot(finercut_70b_param, finercut_70b_c4_ppl, color='purple', label='FinerCut* (Layer)')
axes[2, 0].plot(slicegpt_70b_param, slicegpt_70b_c4_ppl, color='orange', label='SliceGPT (Channel)')
axes[2, 0].plot(sleb_70b_param, sleb_70b_c4_ppl, color='brown', label='SLEB (Block)')
axes[2, 0].plot(our_block_70b_param, our_block_70b_c4_ppl, color='teal', label='Our Block')
axes[2, 0].plot(our_layer_70b_param, our_layer_70b_c4_ppl, color='blue', label='Our Layer')
axes[2, 0].set_title(f'Llama-2-70b')
axes[2, 0].set_xlabel('Params')
axes[2, 0].set_ylabel('C4 Perplexity (PPL)')
axes[2, 0].set_xticks(np.array(range(5, 10)) / 10)
axes[2, 0].set_yticks(np.array(range(5, 25, 5)))
axes[2, 0].set_ylim([5, 22.5])
axes[2, 0].grid(c='0.8')
# axes[2, 0].set_ylim([5, 25])
axes[2, 0].legend(loc="upper right")

axes[2, 1].plot(finercut_70b_param, finercut_70b_w2_ppl, color='purple', label='FinerCut* (Layer)')
# axes[2, 1].plot(slicegpt_70b_param, slicegpt_70b_w2_ppl, color='orange', label='SliceGPT (Channel)')
axes[2, 1].plot(sleb_70b_param, sleb_70b_w2_ppl, color='brown', label='SLEB (Block)')
axes[2, 1].plot(our_block_70b_param, our_block_70b_w2_ppl, color='teal', label='Our Block')
axes[2, 1].plot(slicegpt_70b_param, our_layer_70b_w2_ppl, color='blue', label='Our Layer')
axes[2, 1].set_title(f'Llama-2-70b')
axes[2, 1].set_xlabel('Params')
axes[2, 1].set_ylabel('Wikitext2 Perplexity (PPL)')
axes[2, 1].set_xticks(np.array(range(5, 10)) / 10)
axes[2, 1].set_yticks(np.array(range(0, 25, 5)))
axes[2, 1].set_ylim([2, 16])
axes[2, 1].grid(c='0.8')
axes[2, 1].legend(loc="upper right")


axes[2, 2].plot(finercut_70b_param, finercut_70b_zeroshot, color='purple', label='FinerCut* (Layer)')
axes[2, 2].plot(slicegpt_70b_param, slicegpt_70b_zeroshot, color='orange', label='SliceGPT (Channel)')
axes[2, 2].plot(sleb_70b_param, sleb_70b_zeroshot, color='brown', label='SLEB (Block)')
axes[2, 2].plot(our_block_70b_param, our_block_70b_zeroshot, color='teal', label='Our Block')
axes[2, 2].plot(our_layer_70b_param, our_layer_70b_zeroshot, color='blue', label='Our Layer')
axes[2, 2].set_title(f'Llama-2-70b')
axes[2, 2].set_xlabel('Params')
axes[2, 2].set_ylabel('Zeroshot task Mean Acc.')
axes[2, 2].set_xticks(np.array(range(5, 10)) / 10)
axes[2, 2].grid(c='0.8')
# axes[0, 2].set_ylim([None, 50])
axes[2, 2].legend(loc="upper left")

fig.tight_layout() 
fig.subplots_adjust(top=0.95)
# plt.figlegend(lines, labels, loc = 'lower center', ncol=5, labelspacing=0.)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=len(labels), labelspacing=0., fontsize=15)
plt.show()
plt.savefig(sleb_figure, dpi=300)