import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from datetime import datetime

now = datetime.now() # current date and time
date_time = now.strftime("%Y%m%d%H%M%S")

sleb_figure = f'fig/layer/{date_time}_total.png'


sleb_128_7b_w2_ppl = "6.609770298	7.41885376	8.384570122	10.12059593	12.12338829	15.51503277	22.23426819	35.61605072	49.133358"
sleb_128_7b_w2_ppl = list(map(float, sleb_128_7b_w2_ppl.split("\t")))
sleb_128_7b_c4_ppl = "8.728996277	9.672170639	10.96531582	12.69634533	14.77611828	18.53191185	23.92844009	38.66782379	51.02135468"
sleb_128_7b_c4_ppl = list(map(float, sleb_128_7b_c4_ppl.split("\t")))
sleb_128_7b_param = "0.89621114	0.854598446	0.80246114	0.750323834	0.698186528	0.646049223	0.604436528	0.541774611	0.500161917"
sleb_128_7b_param = list(map(float, sleb_128_7b_param.split("\t")))
sleb_128_7b_latency = "4.447615644	4.155571699	3.954386498	3.72502402	3.506142619	3.301202521	3.025111539	2.855957752	2.596383587"
sleb_128_7b_latency = list(map(float, sleb_128_7b_latency.split("\t")))

nsga_7b_w2_ppl = "6.419158459	6.954999924	7.932456017	9.635126114	12.04807854	16.91271782	20.24876022	27.00183678	39.60480881"
nsga_7b_w2_ppl = list(map(float, nsga_7b_w2_ppl.split("\t")))
nsga_7b_c4_ppl = "8.563390732	9.354728699	10.60588646	12.19474888	14.35938168	19.05661583	23.14393425	30.93745804	42.01548004"
nsga_7b_c4_ppl = list(map(float, nsga_7b_c4_ppl.split("\t")))
nsga_7b_param = "0.896	0.8544	0.8023	0.7503	0.6982	0.646	0.6043	0.5521	0.5"
nsga_7b_param = list(map(float, nsga_7b_param.split("\t")))
nsga_7b_latency = "4.65	4.25	4.06	3.71	3.49	3.32	3.12	2.92	2.7"
nsga_7b_latency = list(map(float, nsga_7b_latency.split("\t")))

sleb_7b_w2_ppl = "6.468054771	7.458059311	8.110439301	10.39372253	13.82021713	17.36932755	29.92934036	43.75978088	106.2018356"
sleb_7b_w2_ppl = list(map(float, sleb_7b_w2_ppl.split("\t")))
sleb_7b_c4_ppl = "8.710828781	10.05723381	10.89597225	13.73572254	17.41545868	21.86510086	32.05850983	40.25984955	85.97419739"
sleb_7b_c4_ppl = list(map(float, sleb_7b_c4_ppl.split("\t")))
sleb_7b_param = "0.90625	0.84375	0.8125	0.75	0.6875	0.65625	0.59375	0.5625	0.5"
sleb_7b_param = list(map(float, sleb_7b_param.split("\t")))
# sleb_7b_latency = "4.65	4.25	4.06	3.71	3.49	3.32	3.12	2.92	2.7"
# sleb_7b_latency = list(map(float, sleb_7b_latency.split("\t")))

slicegpt_param = [0.9, 0.8, 0.7, 0.6, 0.5]
slicegpt_c4_ppl = [17.3134, 25.8254, 41.2653, 72.8173, 137.7248]
slicegpt_w2_ppl = [5.9564, 6.8606, 8.6269, 12.802, 21.0916]
slicegpt_latency = [5.772, 5.265, 4.8636, 4.711, 4.139]

flap_param = [0.9, 0.8, 0.7, 0.6, 0.5]
flap_c4_ppl = [8.623231888, 11.29648495, 15.55335712, 25.51643372, 52.63236237]
flap_w2_ppl = [6.084458828, 7.451695919, 10.56966591, 19.85195923, 54.05876923]
flap_latency = [5.087628107, 4.818382587, 4.320798185, 3.788519752, 3.425058297]

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
fig.subplots_adjust(hspace=0.5, wspace=0.3)

# axes[0, 0].scatter(sleb_128_7b_sparsity, sleb_128_7b_ppl, color='gray', s=3, label='sleb layer 128 (20m)', alpha=0.5)
# axes[0, 0].scatter(sleb_128_7b_param, sleb_128_7b_c4_ppl, color='lightgray', s=3, label='sleb layer 128 (20m)')
axes[0, 0].plot(sleb_128_7b_param, sleb_128_7b_c4_ppl, color='lightgray', label='sleb layer 128 (20m)')
# axes[0, 0].scatter(sleb_256_7b_param, sleb_256_7b_ppl, color='red', s=3, label='sleb layer 256 (48m)')
# axes[0, 0].scatter(sleb_512_7b_param, sleb_512_7b_ppl, color='purple', s=3, label='sleb layer 512 (78m)')
# axes[0, 0].scatter(nsga_sparsity, nsga_ppl, color='purple', s=3, label='nsga (77m)', alpha=0.5)
# axes[0, 0].scatter(nsga_param, nsga_c4_ppl, color='blue', s=3, label='nsga (77m)')
axes[0, 0].plot(nsga_7b_param, nsga_7b_c4_ppl, color='blue', label='nsga (77m)')
axes[0, 0].plot(slicegpt_param, slicegpt_c4_ppl, color='orange', label='slicegpt')
axes[0, 0].plot(flap_param, flap_c4_ppl, color='lightgreen', label='flap')
axes[0, 0].plot(sleb_7b_param, sleb_7b_c4_ppl, color='violet', label='sleb')
axes[0, 0].set_title(f'Llama-2-7b c4')
axes[0, 0].set_xlabel('Params')
axes[0, 0].set_ylabel('PPL')
axes[0, 0].set_ylim([None, 55])
axes[0, 0].legend(loc="upper right")


axes[0, 1].plot(sleb_128_7b_latency, sleb_128_7b_c4_ppl, color='lightgray', label='sleb layer 128 (20m)')
# axes[0, 1].scatter(sleb_256_7b_param, sleb_256_7b_ppl, color='red', s=3, label='sleb layer 256 (48m)')
# axes[0, 1].scatter(sleb_512_7b_param, sleb_512_7b_ppl, color='purple', s=3, label='sleb layer 512 (78m)')
axes[0, 1].plot(nsga_7b_latency, nsga_7b_c4_ppl, color='blue', label='nsga (77m)')
axes[0, 1].plot(slicegpt_latency, slicegpt_c4_ppl, color='orange', label='slicegpt')
axes[0, 1].plot(flap_latency, flap_c4_ppl, color='lightgreen', label='flap')
axes[0, 1].set_title(f'Llama-2-7b c4')
axes[0, 1].set_xlabel('Latency')
axes[0, 1].set_ylabel('PPL')
axes[0, 1].set_ylim([None, 55])
axes[0, 1].legend(loc="upper right")


# axes[0, 0].scatter(sleb_128_7b_sparsity, sleb_128_7b_w2_ppl, color='gray', s=3, label='sleb layer 128 (20m)', alpha=0.5)
axes[0, 2].plot(sleb_128_7b_param, sleb_128_7b_w2_ppl, color='lightgray', label='sleb layer 128 (20m)')
# axes[0, 0].scatter(sleb_256_7b_param, sleb_256_7b_ppl, color='red', s=3, label='sleb layer 256 (48m)')
# axes[0, 0].scatter(sleb_512_7b_param, sleb_512_7b_ppl, color='purple', s=3, label='sleb layer 512 (78m)')
# axes[0, 0].scatter(nsga_sparsity, nsga_ppl, color='purple', s=3, label='nsga (77m)', alpha=0.5)
axes[0, 2].plot(nsga_7b_param, nsga_7b_w2_ppl, color='blue', label='nsga (77m)')
axes[0, 2].plot(slicegpt_param, slicegpt_w2_ppl, color='orange', label='slicegpt')
axes[0, 2].plot(flap_param, flap_w2_ppl, color='lightgreen', label='flap')
axes[0, 2].plot(sleb_7b_param, sleb_7b_w2_ppl, color='violet', label='sleb')
axes[0, 2].set_title(f'Llama-2-7b wikitext2')
axes[0, 2].set_xlabel('Params')
axes[0, 2].set_ylabel('PPL')
axes[0, 2].set_ylim([None, 50])
axes[0, 2].legend(loc="upper right")


axes[0, 3].plot(sleb_128_7b_latency, sleb_128_7b_w2_ppl, color='lightgray', label='sleb layer 128 (20m)')
# axes[0, 1].scatter(sleb_256_7b_param, sleb_256_7b_ppl, color='red', s=3, label='sleb layer 256 (48m)')
# axes[0, 1].scatter(sleb_512_7b_param, sleb_512_7b_ppl, color='purple', s=3, label='sleb layer 512 (78m)')
axes[0, 3].plot(nsga_7b_latency, nsga_7b_w2_ppl, color='blue', label='nsga (77m)')
axes[0, 3].plot(slicegpt_latency, slicegpt_w2_ppl, color='orange', label='slicegpt')
axes[0, 3].plot(flap_latency, flap_w2_ppl, color='lightgreen', label='flap')
axes[0, 3].set_title(f'Llama-2-7b wikitext2')
axes[0, 3].set_xlabel('Latency')
axes[0, 3].set_ylabel('PPL')
axes[0, 3].set_ylim([None, 50])
axes[0, 3].legend(loc="upper right")


sleb_128_13b_w2_ppl = "5.410739899	5.848044872	6.434978008	7.254586697	8.478711128	9.871990204	12.59064102	18.04015923	32.46655655"
sleb_128_13b_w2_ppl = list(map(float, sleb_128_13b_w2_ppl.split("\t")))
sleb_128_13b_c4_ppl = "7.459181786	8.07357502	8.860198021	9.893856049	11.39112663	12.91997337	15.32797527	19.59002876	35.91783524"
sleb_128_13b_c4_ppl = list(map(float, sleb_128_13b_c4_ppl.split("\t")))
sleb_128_13b_param = "0.900826446	0.850826446	0.800826446	0.750826446	0.692355372	0.650619835	0.600619835	0.550619835	0.49214876"
sleb_128_13b_param = list(map(float, sleb_128_13b_param.split("\t")))
sleb_128_13b_latency = "7.504453059	7.056995951	6.614486172	6.147576247	5.750985445	5.444466345	5.022227337	4.565828106	4.154034714"
sleb_128_13b_latency = list(map(float, sleb_128_13b_latency.split("\t")))

nsga_13b_w2_ppl = "5.385161877	5.793252945	6.329028606	7.036133289	8.182165146	9.471266747	11.52717876	15.5002718	20.820261"
nsga_13b_w2_ppl = list(map(float, nsga_13b_w2_ppl.split("\t")))
nsga_13b_c4_ppl = "7.4330616	8.024552345	8.770359993	9.765916824	10.87189388	12.87720966	14.5889864	18.97055054	23.86247826"
nsga_13b_c4_ppl = list(map(float, nsga_13b_c4_ppl.split("\t")))
nsga_13b_param = "0.9006	0.8506	0.8006	0.7506	0.7006	0.6504	0.6004	0.5504	0.5002"
nsga_13b_param = list(map(float, nsga_13b_param.split("\t")))
nsga_13b_latency = "7.63	7.27	6.7	6.42	5.88	5.56	5.12	4.79	4.43"
nsga_13b_latency = list(map(float, nsga_13b_latency.split("\t")))

slicegpt_param = [0.9, 0.8, 0.7, 0.6, 0.5]
slicegpt_w2_ppl = [5.2941, 6.0389, 7.4356, 10.6064, 17.5747]
slicegpt_c4_ppl = [15.509, 23.7214, 38.4982, 67.2131, 121.7989]
slicegpt_latency = [9.2124, 8.7407, 7.9494, 7.3503, 7.1742]

flap_param = [0.9, 0.8, 0.7, 0.6, 0.5]
flap_w2_ppl = [5.43, 6.27, 7.70, 11.49, 19.52]
flap_c4_ppl = [7.79, 9.65, 12.26, 17.74, 27.44]
flap_latency = [8.789877715, 8.15873653, 7.382896387, 6.396090912, 5.435466782]

sleb_13b_w2_ppl = "5.385161877	5.793252945	6.329028606	7.036133289	8.182165146	9.471266747	11.52717876	15.5002718	20.820261"
sleb_13b_w2_ppl = list(map(float, sleb_13b_w2_ppl.split("\t")))
sleb_13b_c4_ppl = "7.4330616	8.024552345	8.770359993	9.765916824	10.87189388	12.87720966	14.5889864	18.97055054	23.86247826"
sleb_13b_c4_ppl = list(map(float, sleb_13b_c4_ppl.split("\t")))
sleb_13b_param = "0.9006	0.8506	0.8006	0.7506	0.7006	0.6504	0.6004	0.5504	0.5002"
sleb_13b_param = list(map(float, sleb_13b_param.split("\t")))
sleb_13b_latency = "7.63	7.27	6.7	6.42	5.88	5.56	5.12	4.79	4.43"
sleb_13b_latency = list(map(float, sleb_13b_latency.split("\t")))

axes[1, 0].plot(sleb_128_13b_param, sleb_128_13b_c4_ppl, color='lightgray', label='sleb layer 128 (55m)')
# axes[1, 0].scatter(sleb_256_13b_param, sleb_256_13b_ppl, color='red', s=3, label='sleb layer 256 (138m)')
# axes[1, 0].scatter(sleb_512_13b_param, sleb_512_13b_ppl, color='purple', s=3, label='sleb layer 512 (246m)')
axes[1, 0].plot(nsga_13b_param, nsga_13b_c4_ppl, color='blue', label='nsga 128 (210m)')
axes[1, 0].plot(slicegpt_param, slicegpt_c4_ppl, color='orange', label='slicegpt')
axes[1, 0].plot(flap_param, flap_c4_ppl, color='lightgreen', label='flap')
axes[1, 0].set_title(f'Llama-2-13b c4')
axes[1, 0].set_xlabel('Params')
axes[1, 0].set_ylabel('PPL')
axes[1, 0].set_ylim([5, 30])
axes[1, 0].legend(loc="upper right")

axes[1, 1].plot(sleb_128_13b_latency, sleb_128_13b_c4_ppl, color='lightgray', label='sleb layer 128 (20m)')
# axes[0, 1].scatter(sleb_256_7b_param, sleb_256_7b_ppl, color='red', s=3, label='sleb layer 256 (48m)')
# axes[0, 1].scatter(sleb_512_7b_param, sleb_512_7b_ppl, color='purple', s=3, label='sleb layer 512 (78m)')
axes[1, 1].plot(nsga_13b_latency, nsga_13b_c4_ppl, color='blue', label='nsga (77m)')
axes[1, 1].plot(slicegpt_latency, slicegpt_c4_ppl, color='orange', label='slicegpt')
axes[1, 1].plot(flap_latency, flap_c4_ppl, color='lightgreen', label='flap')
axes[1, 1].set_title(f'Llama-2-13b')
axes[1, 1].set_xlabel('Latency')
axes[1, 1].set_ylabel('PPL')
# axes[0, 1].set_xlim([0.8, 1.0])
axes[1, 1].set_ylim([5, 30])
axes[1, 1].legend(loc="upper right")


# axes[1, 0].scatter(sleb_128_7b_sparsity, sleb_128_7b_w2_ppl, color='gray', s=3, label='sleb layer 128 (20m)', alpha=0.5)
axes[1, 2].plot(sleb_128_13b_param, sleb_128_13b_w2_ppl, color='gray', label='sleb layer 128 (20m)')
# axes[0, 0].scatter(sleb_256_7b_param, sleb_256_7b_ppl, color='red', s=3, label='sleb layer 256 (48m)')
# axes[0, 0].scatter(sleb_512_7b_param, sleb_512_7b_ppl, color='purple', s=3, label='sleb layer 512 (78m)')
# axes[0, 0].scatter(nsga_sparsity, nsga_ppl, color='purple', s=3, label='nsga (77m)', alpha=0.5)
axes[1, 2].plot(nsga_13b_param, nsga_13b_w2_ppl, color='blue', label='nsga (77m)')
# axes[1, 2].plot(slicegpt_param, slicegpt_w2_ppl, color='orange', label='slicegpt')
# axes[1, 2].plot(flap_param, flap_w2_ppl, color='lightgreen', label='flap')
axes[1, 2].set_title(f'Llama-2-13b wikitext2')
axes[1, 2].set_xlabel('Params')
axes[1, 2].set_ylabel('PPL')
axes[1, 2].set_ylim([None, 35])
axes[1, 2].legend(loc="upper right")


axes[1, 3].plot(sleb_128_13b_latency, sleb_128_13b_w2_ppl, color='lightgray', label='sleb layer 128 (20m)')
axes[1, 3].plot(nsga_13b_latency, nsga_13b_w2_ppl, color='blue', label='nsga (77m)')
# axes[1, 3].plot(slicegpt_latency, slicegpt_w2_ppl, color='orange', label='slicegpt')
# axes[1, 3].plot(flap_latency, flap_w2_ppl, color='lightgreen', label='flap')
axes[1, 3].set_title(f'Llama-2-13b')
axes[1, 3].set_xlabel('Latency')
axes[1, 3].set_ylabel('PPL')
# axes[0, 1].set_xlim([0.8, 1.0])
axes[1, 3].set_ylim([None, 35])
axes[1, 3].legend(loc="upper right")

plt.show()
plt.savefig(sleb_figure, dpi=300)