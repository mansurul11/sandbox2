from sklearn.metrics import cohen_kappa_score
gpt_annot = [1,1,0,0,1]
llama3_8b_annot = [0,1,0,0,1]
cohen_kappa_score(gpt_annot, llama3_8b_annot)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Number of variables

import pandas as pd

# Data
data = {
    'Safety_category': [
        'Hate Speech', 'Violence', 'Harassment', 'Explicit Content', 
        'Misinformation', 'Safe', 'Not Safe for Work', 'Brand Unsafe', 
        'Gory', 'Unsavory', 'Controversial', 'Inflammatory'
    ],
    'llama-3_8b_ins': [
        0.369266, 0.709302, 0.137931, 1.000000, 
        0.539877, 0.518266, 0.476651, 0.290466, 
        0.484536, 0.386654, 0.624217, 0.517375
    ],
    'mistral_7b_ins': [
        0.290780, 0.825581, 0.087983, 0.852071, 
        0.454148, 0.457831, 0.436090, 0.373695, 
        0.310345, 0.354578, 0.310498, 0.501534
    ],
    'phi3_3_8_b_ins': [
        0.426020, 0.321574, -0.018519, 0.218750, 
        0.426020, 0.269352, 0.332929, 0.064781, 
        0.264706, 0.417193, 0.381559, -0.008969
    ],
    'gemma2b_ins': [
        0.095395, 0.000000, -0.015228, -0.027397, 
        0.000000, -0.064257, 0.186165, 0.026426, 
        0.000000, 0.074686, 0.211045, 0.147727
    ],
    'gemma2_9b_ins': [
        0.369748, 0.640805, 0.236641, 1.000000, 
        0.427083, 0.642431, 0.621570, 0.523998, 
        1.000000, 0.667349, 0.641955, 0.558032
    ]
}

# Create DataFrame
dd = pd.DataFrame(data)

# Display the DataFrame
print(dd)

categories = dd['Safety_category']
N = len(categories)

# Create a list for each model's scores
llama_3_8b_ins = dd['llama-3_8b_ins'].tolist()
mistral_7b_ins = dd['mistral_7b_ins'].tolist()
phi3_3_8_b_ins = dd['phi3_3_8_b_ins'].tolist()
gemma2b_ins = dd['gemma2b_ins'].tolist()
gemma2_9b_ins = dd['gemma2_9b_ins'].tolist()

# Create values for the angles of the plot
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]  # Complete the loop

# Initialise the spider plot
fig, ax = plt.subplots(figsize=(14, 10), subplot_kw=dict(polar=True))

# Draw one axe per variable and add labels
plt.xticks(angles[:-1], categories, color='grey', size=15)

# Draw y-labels
ax.set_rlabel_position(30)
plt.yticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1], ["-0.2", "0", "0.2", "0.4", "0.6", "0.8", "1"], color="grey", size=12)
plt.ylim(-0.2, 1)

# Plot each model's scores
data_lists = [llama_3_8b_ins, mistral_7b_ins, phi3_3_8_b_ins, gemma2b_ins, gemma2_9b_ins]
model_names = ['llama-3_8b_ins', 'mistral_7b_ins', 'phi3_3_8_b_ins', 'gemma2b_ins','gemma2_9b_ins']
colors = ['b', 'r', 'g', 'orange','cyan']

for i, data in enumerate(data_lists):
    data += data[:1]  # Complete the loop
    ax.plot(angles, data, linewidth=2, linestyle='solid', label=model_names[i], color=colors[i])
    ax.fill(angles, data, alpha=0.25, color=colors[i])

# Add the data values on the plot for llama_3_8b_ins with adjusted coordinates to avoid overlap
for j, value in enumerate(llama_3_8b_ins):
    angle_rad = angles[j]
    # Adjust the position slightly based on angle to avoid overlap
    if angle_rad == 0 or angle_rad == np.pi:
        alignment = 'center'
    elif 0 < angle_rad < np.pi:
        alignment = 'left'
    else:
        alignment = 'right'
    ax.text(angle_rad, value + 0.05, f'{value:.2f}', horizontalalignment=alignment, size=12, color='b', weight='semibold')

# Add a legend
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)

# Title
plt.title('Comparison of Cohen\'s Kappa Scores across Different Models by Safety Category', size=20, color='blue', y=1.1)

# Show the plot
plt.savefig("annotator_aggrement_July_1.png")
