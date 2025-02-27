# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# # Helper Functions
# def load_config(config_path, model_name):
#     """Load model configuration from a JSON file."""
#     with open(config_path, 'r') as f:
#         return json.load(f)[model_name]

# def load_iteration_data(arch_folder, iteration):
#     """Load architecture data for a specific iteration."""
#     file_path = os.path.join(arch_folder, f'iter_{iteration}.stats')
#     with open(file_path, 'r') as f:
#         return json.load(f)

# def process_candidates(candidates):
#     """Process candidate architectures to extract self_attn.v_proj."""
#     return np.array([c['linear']['self_attn.k_proj'] for c in candidates])

# def visualize_top5_and_mean(top5_data, mean_data, model_name, fig_path):
#     """Visualize the top 5 and their mean in a 2x3 subplot layout."""
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))

#     # for idx, data in enumerate(top5_data):
#     for idx in range(5):
#         data = top5_data[:, idx, :]
#         ax = axes[idx // 3, idx % 3]
#         im = ax.matshow(data.T, aspect='auto')
#         ax.set_title(f"Top {idx + 1}")
#         ax.set_xlabel('Iteration')
#         ax.set_ylabel('Linear Index')
#         fig.colorbar(im, ax=ax, location='right', shrink=0.6)

#     # Plot the mean in the last subplot
#     ax = axes[-1, -1]
#     im = ax.matshow(mean_data.T, aspect='auto')
#     ax.set_title("Mean of Top 5")
#     ax.set_xlabel('Iteration')
#     ax.set_ylabel('Linear Index')
#     fig.colorbar(im, ax=ax, location='right', shrink=0.6)

#     # Save the final visualization
#     plt.tight_layout()
#     plt.savefig(fig_path, dpi=300)
#     plt.close()
#     print(f"Visualization saved at: {fig_path}")

# # Main Script
# def main():
#     model_name = 'Llama-2-7b-hf'
#     arch_folder = f'/NAS/SJ/nsgaquant/save/search/2411270816_{model_name}_bits_loss_hqq_iter_300_nsga2_234_obj_2_4_jsd_mut_0.1_layer_prune_1.0_1.0'
#     iter_list = range(1, 300)  # Specify iterations to process
#     fig_path = '/NAS/Woo/Automation/autoopt/visualize_result/final_top5.png'
#     config_path = 'config/llama.json'

#     # Load model configuration
#     config = load_config(config_path, model_name)

#     top5_data = []
#     for iteration in iter_list:
#         # Load data for the current iteration
#         arch_data = load_iteration_data(arch_folder, iteration)
        
#         # Extract top 5 candidates based on some criterion (e.g., second-to-last metric)
#         sorted_candidates = sorted(arch_data['candidates'], key=lambda x: x[-2])[:5]
        
#         # Process candidates to extract relevant data
#         processed_data = process_candidates([c[0] for c in sorted_candidates])

#         # Accumulate top 5 data for visualization
#         top5_data.append(processed_data)

#     # Convert to numpy array for easier manipulation
#     top5_data = np.array(top5_data)  # Shape: (iterations, 5, linear_size)
    
#     # Compute the mean across top 5
#     mean_data = top5_data.mean(axis=1)  # Average over top 5 for each iteration

#     # Visualize top 5 and mean
#     visualize_top5_and_mean(top5_data, mean_data, model_name, fig_path)

# if __name__ == "__main__":
#     main()

# import os
# import json
# import numpy as np
# import matplotlib.pyplot as plt

# # Helper Functions
# def load_config(config_path, model_name):
#     """Load model configuration from a JSON file."""
#     with open(config_path, 'r') as f:
#         return json.load(f)[model_name]

# def load_iteration_data(arch_folder, iteration):
#     """Load architecture data for a specific iteration."""
#     file_path = os.path.join(arch_folder, f'iter_{iteration}.stats')
#     with open(file_path, 'r') as f:
#         return json.load(f)

# def extract_relevant_data(candidate):
#     """Extract q_proj, k_proj, v_proj, o_proj, up_proj, gate_proj, down_proj."""
#     linear = candidate['linear']
#     return [
#         linear['self_attn.q_proj'],
#         linear['self_attn.k_proj'],
#         linear['self_attn.v_proj'],
#         linear['self_attn.o_proj'],
#         linear['mlp.up_proj'],
#         linear['mlp.gate_proj'],
#         linear['mlp.down_proj'],
#     ]

# def visualize_candidates_and_matrix(results, last_matrix, model_name, fig_path):
#     """Visualize 2x4 grid with candidates and final 32x7 matrix."""
#     fig, axes = plt.subplots(2, 4, figsize=(15, 10))

#     # Plot extracted data
#     for idx in range(7):
#         data = results[:, idx, :]
#         ax = axes[idx // 4, idx % 4]
#         im = ax.matshow(data.T, aspect='auto')
#         ax.set_title(f"{['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj'][idx]}")
#         ax.set_xlabel("Iteration")
#         ax.set_ylabel("Linear Index")
#         fig.colorbar(im, ax=ax, location='right', shrink=0.6)

#     # Plot the last matrix in the final subplot
#     ax = axes[-1, -1]
#     im = ax.matshow(last_matrix, aspect='auto')
#     ax.set_title("Final Iteration All Linear Matrix")
#     ax.set_xlabel("q, k, v, o, up, gate, down")
#     ax.set_ylabel("Entries")
#     fig.colorbar(im, ax=ax, location='right', shrink=0.6)

#     # Save the visualization
#     plt.tight_layout()
#     plt.savefig(fig_path, dpi=300)
#     plt.close()
#     print(f"Visualization saved at: {fig_path}")

# # Main Script
# def main():
#     model_name = 'Llama-2-7b-hf'
#     arch_folder = f'/NAS/SJ/nsgaquant/save/search/2411270816_{model_name}_bits_loss_hqq_iter_300_nsga2_234_obj_2_4_jsd_mut_0.1_layer_prune_1.0_1.0'
#     iter_list = range(1, 300)  # Specify iterations to process
#     min_bit = 2.8
#     max_bit = 3.2
#     fig_path = f'/NAS/Woo/Automation/autoopt/visualize_result/final_output_{min_bit}~{max_bit}.png'
#     config_path = 'config/llama.json'

#     # Load model configuration
#     config = load_config(config_path, model_name)

#     extracted_data = []
#     final_iteration_matrix = None

#     for iteration in iter_list:
#         # Load data for the current iteration
#         arch_data = load_iteration_data(arch_folder, iteration)

#         # Filter candidates with condition 2 < c[1] <= 2.25
#         filtered_candidates = [c for c in arch_data['candidates'] if min_bit < c[2] <= max_bit]

#         if filtered_candidates:
#             # Select candidate with the lowest c[2]
#             best_candidate = min(filtered_candidates, key=lambda x: x[1])
#             extracted = extract_relevant_data(best_candidate[0])
#             extracted_data.append(np.array(extracted))

#         # If this is the last iteration, prepare the final matrix
#     else:
#         final_iteration_matrix = np.array(extracted).T

#     # If no valid candidates are found, skip visualization
#     if not extracted_data:
#         print("No candidates matched the criteria. Visualization skipped.")
#         return

#     extracted_data = np.array(extracted_data)
#     # Create the visualization
#     visualize_candidates_and_matrix(
#         extracted_data,  # First 7 iterations for the 2x4 layout
#         final_iteration_matrix,  # Final matrix for the last subplot
#         model_name,
#         fig_path,
#     )

# if __name__ == "__main__":
#     main()


import os
import json
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# Helper Functions
def load_config(config_path, model_name):
    """Load model configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)[model_name]

def load_iteration_data(arch_folder, iteration):
    """Load architecture data for a specific iteration."""
    file_path = os.path.join(arch_folder, f'iter_{iteration}.stats')
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_relevant_data(candidates):
    """Extract q_proj, k_proj, v_proj, o_proj, up_proj, gate_proj, down_proj."""
    # linear = candidate['linear']
    # return np.concatenate([x for x in [linear['self_attn.q_proj'], linear['self_attn.k_proj'], linear['self_attn.v_proj'], linear['self_attn.o_proj'], linear['mlp.up_proj'], linear['mlp.gate_proj'], linear['mlp.down_proj']]])

    return np.mean([np.array([
        candidate[0]['linear']['self_attn.q_proj'],
        candidate[0]['linear']['self_attn.k_proj'],
        candidate[0]['linear']['self_attn.v_proj'],
        candidate[0]['linear']['self_attn.o_proj'],
        candidate[0]['linear']['mlp.up_proj'],
        candidate[0]['linear']['mlp.gate_proj'],
        candidate[0]['linear']['mlp.down_proj']
    ]) for candidate in candidates], axis=0)

def visualize_candidates_and_matrix(results, last_matrix, model_name, fig_path):
    """Visualize 2x4 grid with candidates and final 32x7 matrix."""
    fig, axes = plt.subplots(2, 4, figsize=(15, 10))

    # Plot extracted data
    for idx in range(7):
        data = results[:, idx, :]
        ax = axes[idx // 4, idx % 4]
        im = ax.matshow(data.T, aspect='auto')
        ax.set_title(f"{['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj'][idx]}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Linear Index")
        fig.colorbar(im, ax=ax, location='right', shrink=0.6)

    # Plot the last matrix in the final subplot
    ax = axes[-1, -1]
    im = ax.matshow(last_matrix, aspect='auto')
    ax.set_title("Final Iteration All Linear Matrix")
    ax.set_xlabel("q, k, v, o, up, gate, down")
    ax.set_ylabel("Entries")
    fig.colorbar(im, ax=ax, location='right', shrink=0.6)

    # Save the visualization
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Visualization saved at: {fig_path}")

# Main Script
def main():
    model_name = 'Llama-2-7b-hf'
    arch_folder = f'/NAS/SJ/nsgaquant/save/search/2411270816_{model_name}_bits_loss_hqq_iter_300_nsga2_234_obj_2_4_jsd_mut_0.1_layer_prune_1.0_1.0'
    iter_list = range(1, 300)  # Specify iterations to process
    min_bit = 2
    max_bit = 2.25
    fig_path = f'/NAS/Woo/Automation/autoopt/visualize_result/nsga_result_analysis/greedy_comparing_ratio.txt'
    config_path = 'config/llama.json'
    ## 4->2비트로 변경한 순서 (별로 안 중요한 순서라 보면 됨)
    desc_greedy = '28.self_attn.q_proj,31.self_attn.k_proj,27.self_attn.q_proj,31.self_attn.q_proj,21.self_attn.k_proj,1.self_attn.k_proj,2.self_attn.k_proj,0.mlp.up_proj,26.self_attn.q_proj,18.self_attn.k_proj,0.mlp.gate_proj,30.self_attn.k_proj,3.self_attn.k_proj,24.self_attn.k_proj,29.self_attn.q_proj,26.self_attn.k_proj,30.self_attn.q_proj,25.self_attn.q_proj,20.self_attn.k_proj,22.self_attn.k_proj,23.self_attn.q_proj,24.self_attn.q_proj,5.self_attn.k_proj,19.self_attn.k_proj,21.self_attn.q_proj,22.self_attn.q_proj,29.self_attn.k_proj,20.self_attn.q_proj,17.self_attn.q_proj,27.self_attn.k_proj,4.self_attn.q_proj,25.self_attn.k_proj,28.self_attn.k_proj,0.mlp.down_proj,13.self_attn.k_proj,16.self_attn.k_proj,0.self_attn.k_proj,26.self_attn.o_proj,30.self_attn.v_proj,15.self_attn.k_proj,16.self_attn.q_proj,18.self_attn.q_proj,19.self_attn.q_proj,7.self_attn.q_proj,8.self_attn.k_proj,14.self_attn.k_proj,28.self_attn.o_proj,23.self_attn.k_proj,1.self_attn.q_proj,4.self_attn.k_proj,9.self_attn.q_proj,25.self_attn.o_proj,2.self_attn.q_proj,29.self_attn.v_proj,17.self_attn.k_proj,6.self_attn.q_proj,23.self_attn.o_proj,30.self_attn.o_proj,21.self_attn.o_proj,5.self_attn.q_proj,15.self_attn.q_proj,12.self_attn.k_proj,8.self_attn.q_proj,27.self_attn.o_proj,14.self_attn.q_proj,13.self_attn.q_proj,22.self_attn.o_proj,10.self_attn.k_proj,9.self_attn.k_proj,31.self_attn.o_proj,1.mlp.gate_proj,11.self_attn.k_proj,6.self_attn.o_proj,7.self_attn.k_proj,12.self_attn.q_proj,20.self_attn.o_proj,0.self_attn.q_proj,29.self_attn.o_proj,6.self_attn.k_proj,2.mlp.gate_proj,24.self_attn.o_proj,19.self_attn.o_proj,28.self_attn.v_proj,3.self_attn.q_proj,4.self_attn.o_proj,10.self_attn.q_proj,18.self_attn.o_proj,11.mlp.gate_proj,31.self_attn.v_proj,23.self_attn.v_proj,19.mlp.gate_proj,19.self_attn.v_proj,11.self_attn.q_proj,17.self_attn.o_proj,0.self_attn.o_proj,10.mlp.gate_proj,7.self_attn.o_proj,3.mlp.gate_proj,1.self_attn.o_proj,13.mlp.gate_proj,14.mlp.gate_proj,12.mlp.gate_proj,13.self_attn.o_proj,26.self_attn.v_proj,25.self_attn.v_proj,26.mlp.down_proj,5.self_attn.o_proj,18.mlp.gate_proj,27.self_attn.v_proj,16.self_attn.o_proj,12.self_attn.o_proj,29.mlp.up_proj,6.mlp.gate_proj,17.mlp.gate_proj,14.self_attn.o_proj,3.self_attn.o_proj,27.mlp.down_proj,21.mlp.gate_proj,1.mlp.up_proj,9.mlp.gate_proj,15.mlp.gate_proj,10.self_attn.o_proj,26.mlp.up_proj,29.mlp.down_proj,5.mlp.gate_proj,25.mlp.down_proj,28.mlp.up_proj,19.mlp.up_proj,16.mlp.gate_proj,23.mlp.up_proj,8.mlp.gate_proj,21.self_attn.v_proj,9.self_attn.o_proj,22.mlp.gate_proj,8.mlp.down_proj,20.self_attn.v_proj,22.self_attn.v_proj,8.self_attn.o_proj,11.mlp.down_proj,24.mlp.down_proj,20.mlp.up_proj,24.mlp.gate_proj,26.mlp.gate_proj,7.mlp.gate_proj,25.mlp.gate_proj,15.self_attn.o_proj,24.self_attn.v_proj,30.mlp.up_proj,11.self_attn.o_proj,14.mlp.up_proj,11.mlp.up_proj,28.mlp.down_proj,18.mlp.up_proj,9.mlp.up_proj,4.mlp.gate_proj,27.mlp.up_proj,18.self_attn.v_proj,23.mlp.down_proj,12.mlp.up_proj,10.mlp.down_proj,31.mlp.up_proj,6.mlp.down_proj,22.mlp.up_proj,25.mlp.up_proj,2.self_attn.o_proj,29.mlp.gate_proj,13.mlp.up_proj,23.mlp.gate_proj,20.mlp.gate_proj,21.mlp.up_proj,22.mlp.down_proj,24.mlp.up_proj,10.mlp.up_proj,31.mlp.gate_proj,30.mlp.down_proj,17.self_attn.v_proj,17.mlp.up_proj,3.mlp.down_proj,16.mlp.up_proj,27.mlp.gate_proj,28.mlp.gate_proj,9.mlp.down_proj,16.self_attn.v_proj,30.mlp.gate_proj,13.mlp.down_proj,15.mlp.down_proj,21.mlp.down_proj,14.mlp.down_proj,18.mlp.down_proj,20.mlp.down_proj,2.mlp.up_proj,12.mlp.down_proj,15.mlp.up_proj,19.mlp.down_proj,7.mlp.down_proj,16.mlp.down_proj,8.mlp.up_proj,17.mlp.down_proj,3.mlp.up_proj,5.mlp.up_proj,7.mlp.up_proj,2.mlp.down_proj,15.self_attn.v_proj,7.self_attn.v_proj,14.self_attn.v_proj,5.mlp.down_proj,4.mlp.up_proj,13.self_attn.v_proj,6.self_attn.v_proj,12.self_attn.v_proj,9.self_attn.v_proj,10.self_attn.v_proj,4.mlp.down_proj,3.self_attn.v_proj,8.self_attn.v_proj,4.self_attn.v_proj,6.mlp.up_proj,5.self_attn.v_proj,11.self_attn.v_proj,2.self_attn.v_proj,31.mlp.down_proj,0.self_attn.v_proj,1.self_attn.v_proj,1.mlp.down_proj'
    desc_bits = '3.994818652849741,3.989637305699482,3.9844559585492227,3.9792746113989637,3.9740932642487046,3.9689119170984455,3.9637305699481864,3.9498056994818653,3.9446243523316062,3.939443005181347,3.925518134715026,3.920336787564767,3.915155440414508,3.909974093264249,3.9047927461139897,3.8996113989637307,3.8944300518134716,3.8892487046632125,3.8840673575129534,3.8788860103626943,3.8737046632124352,3.868523316062176,3.863341968911917,3.858160621761658,3.852979274611399,3.84779792746114,3.8426165803108807,3.8374352331606216,3.8322538860103625,3.8270725388601035,3.8218911917098444,3.8167098445595853,3.8115284974093266,3.797603626943005,3.792422279792746,3.787240932642487,3.782059585492228,3.776878238341969,3.7716968911917097,3.7665155440414506,3.7613341968911915,3.7561528497409324,3.7509715025906734,3.7457901554404147,3.7406088082901556,3.7354274611398965,3.7302461139896375,3.7250647668393784,3.7198834196891193,3.71470207253886,3.709520725388601,3.704339378238342,3.699158031088083,3.693976683937824,3.6887953367875648,3.6836139896373057,3.6784326424870466,3.6732512953367875,3.6680699481865284,3.6628886010362693,3.6577072538860103,3.652525906735751,3.647344559585492,3.642163212435233,3.636981865284974,3.631800518134715,3.6266191709844557,3.621437823834197,3.616256476683938,3.611075129533679,3.5971502590673574,3.5919689119170983,3.5867875647668392,3.58160621761658,3.576424870466321,3.571243523316062,3.566062176165803,3.5608808290155443,3.555699481865285,3.5417746113989637,3.5365932642487046,3.5314119170984455,3.5262305699481864,3.5210492227979273,3.5158678756476682,3.510686528497409,3.50550518134715,3.491580310880829,3.48639896373057,3.481217616580311,3.4672927461139897,3.4621113989637307,3.4569300518134716,3.4517487046632125,3.4465673575129534,3.4326424870466323,3.4274611398963732,3.4135362694300517,3.4083549222797926,3.3944300518134716,3.38050518134715,3.366580310880829,3.36139896373057,3.356217616580311,3.3510362694300517,3.3371113989637307,3.3319300518134716,3.31800518134715,3.312823834196891,3.3076424870466323,3.3024611398963732,3.2885362694300517,3.2746113989637307,3.260686528497409,3.25550518134715,3.250323834196891,3.23639896373057,3.222474093264249,3.2085492227979273,3.1946243523316062,3.180699481865285,3.175518134715026,3.1615932642487046,3.1476683937823835,3.133743523316062,3.119818652849741,3.10589378238342,3.0919689119170983,3.0780440414507773,3.0641191709844557,3.0501943005181347,3.0450129533678756,3.0398316062176165,3.0259067357512954,3.011981865284974,3.006800518134715,3.0016191709844557,2.996437823834197,2.9825129533678756,2.9685880829015545,2.954663212435233,2.940738341968912,2.926813471502591,2.9128886010362693,2.8989637305699483,2.893782383419689,2.88860103626943,2.874676165803109,2.86949481865285,2.8555699481865284,2.8416450777202074,2.827720207253886,2.8137953367875648,2.7998704663212437,2.785945595854922,2.772020725388601,2.766839378238342,2.7529145077720205,2.7389896373056994,2.7250647668393784,2.711139896373057,2.697215025906736,2.6832901554404147,2.669365284974093,2.664183937823834,2.650259067357513,2.6363341968911915,2.6224093264248705,2.6084844559585494,2.594559585492228,2.580634715025907,2.5667098445595853,2.552784974093264,2.538860103626943,2.5249352331606216,2.5197538860103625,2.5058290155440415,2.4919041450777204,2.477979274611399,2.464054404145078,2.4501295336787563,2.4362046632124352,2.431023316062176,2.417098445595855,2.4031735751295336,2.3892487046632125,2.375323834196891,2.36139896373057,2.347474093264249,2.3335492227979273,2.3196243523316062,2.305699481865285,2.2917746113989637,2.2778497409326426,2.263924870466321,2.25,2.236075129533679,2.2221502590673574,2.2082253886010363,2.194300518134715,2.1803756476683938,2.1664507772020727,2.1612694300518136,2.1560880829015545,2.1509067357512954,2.136981865284974,2.123056994818653,2.1178756476683938,2.1126943005181347,2.1075129533678756,2.1023316062176165,2.0971502590673574,2.0832253886010363,2.0780440414507773,2.072862694300518,2.067681347150259,2.053756476683938,2.048575129533679,2.04339378238342,2.0382124352331608,2.0242875647668392,2.01910621761658,2.013924870466321,2.0'
    desc_greedy = desc_greedy.split(',')
    desc_bits = desc_bits.split(',')

    # Load model configuration
    config = load_config(config_path, model_name)

    extracted_data = np.array([0.0 for _ in range(224)])
    final_iteration_matrix = None

    ratio_list = []
    for iteration in iter_list:
        # Load data for the current iteration
        arch_data = load_iteration_data(arch_folder, iteration)

        # Filter candidates with condition 2 < c[1] <= 2.25
        filtered_candidates = [c for c in arch_data['candidates'] if min_bit < c[2] <= max_bit]

        if filtered_candidates:
            # Select candidate with the lowest c[2]
            best_candidate = sorted(filtered_candidates, key=lambda x: x[1])

            mean_bits = sum([x[2] for x in best_candidate]) / len(best_candidate)
            closest_bit = min(desc_bits, key=lambda x: abs(float(x) - mean_bits))
            closest_index = desc_bits.index(closest_bit)
            print(f"Mean bits: {mean_bits}, Closest desc_bits: {closest_bit}, desc_greedy : {desc_greedy[closest_index]}, Index: {closest_index}")
            
            extracted = extract_relevant_data(best_candidate)
            extracted_data = np.concatenate(extracted)
            
            # extracted_data += np.log(iteration) * extracted

        # print(extracted_data)
        # Sort the extracted data in descending order and get the indices
        sorted_indices = np.argsort(extracted_data)
        # # Create a list of all possible words
        words = [f"{i}.self_attn.q_proj" for i in range(32)] + \
            [f"{i}.self_attn.k_proj" for i in range(32)] + \
            [f"{i}.self_attn.v_proj" for i in range(32)] + \
            [f"{i}.self_attn.o_proj" for i in range(32)] + \
            [f"{i}.mlp.up_proj" for i in range(32)] + \
            [f"{i}.mlp.gate_proj" for i in range(32)] + \
            [f"{i}.mlp.down_proj" for i in range(32)]

        # # Sort the words based on the sorted indices
        sorted_words = [words[idx] for idx in sorted_indices]
        # print(sorted_words)

        # print(desc_greedy[closest_index:])
        # print(sorted_words[closest_index:])

        common_linear = set(desc_greedy[closest_index:]) & set(sorted_words[closest_index:])
        # print(f"Common linear: {common_linear}")
        # print(f"Common linear count: {len(common_linear)}")
        print(f"Common linear ratio: {(len(common_linear) - 4) / (len(desc_greedy[closest_index:]) - 4)}")
        ratio_list.append((len(common_linear) - 4) / (len(desc_greedy[closest_index:]) - 4))

        if iteration % 10 == 0:
            import gc; gc.collect()

    with open(fig_path, 'w') as f:
        for ratio in ratio_list:
            f.write(str(ratio) + '\n')

    plt.plot(ratio_list)
    plt.savefig(fig_path.replace('.txt', '.png'), dpi=300)

    # # Create a dictionary to map words to their indices
    # word_to_index = {word: idx for idx, word in enumerate(words)}

    # # Convert desc_greedy to indices
    # desc_greedy_indices = [word_to_index[word] for word in desc_greedy]
    # desc_nsga_indices = [word_to_index[word] for word in sorted_words]
    # # Calculate Spearman rank correlation coefficient
    # min_p_top_k = 0
    # min_p_corr = 1
    # min_p_value = 1
    # for top_k in range(1, 224):
    #     correlation, p_value = spearmanr(desc_greedy_indices[top_k:], desc_nsga_indices[top_k:])
    #     if p_value < min_p_value and p_value > 0:
    #         min_p_top_k = top_k
    #         min_p_corr = correlation
    #         min_p_value = p_value
    #     print(f"{top_k} : Spearman rank correlation coefficient, p_value: {correlation, p_value}")

    # print(f"min_p_top_k: {min_p_top_k}, min_p_corr: {min_p_corr}, min_p_value: {min_p_value}")

    # Print the indices
    # print(desc_greedy_indices)

    # Print the sorted words
    # for word in sorted_words:
    #     print(word)
    # with open(fig_path, 'w') as f:
    #     for word in sorted_words:
    #         f.write(word + '\n')
    # print(f"Sorted words saved at: {fig_path}")

if __name__ == "__main__":
    main()