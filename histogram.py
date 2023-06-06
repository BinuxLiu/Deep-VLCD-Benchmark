import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import gaussian_kde
from scipy.optimize import fsolve

method = "superglue"

# read in data

gt_files = [file for file in os.listdir(f"./output/{method}/") if file.endswith(".txt")]
for gt_file in gt_files:
    data_1 = []
    data_2 = []
    dataset_name = gt_file.split(".")[0]
    dataset_name = "robotcar_qAutumn_dbNight_diff_final"
    print(dataset_name)
    with open(f"./output/{method}/{dataset_name}.txt", 'r') as input:
        for line in input:
            parts = line.strip().split(', ')
            if int(parts[-2]) == 1:
                data_1.append(int(parts[-1]))
            else:
                data_2.append(int(parts[-1]))

    # fit data
    kde_1 = gaussian_kde(data_1)
    kde_2 = gaussian_kde(data_2)

    def difference(x):
        return kde_1(x) - kde_2(x)

    intersection_points = fsolve(difference, 197)
    print("Intersection points:", intersection_points)

    dataset_name = dataset_name.split('robotcar_')[1].split('_final')[0]
    # generate x values
    x = np.linspace(min(min(data_1), min(data_2)), max(max(data_1), max(data_2)), 1000)

    # plot curves
    plt.plot(x, kde_1(x), color='blue')
    plt.plot(x, kde_2(x), color='red')
    plt.xlabel('Number of Matches')
    plt.ylabel('Density')
    plt.title(f'Density Plot of Number of the Matches of {method}')
    # plt.title(f'Density Plot of Number of the Matches on {dataset_name}')
    plt.savefig(f'density_plot_{method}_{dataset_name}.png')
    plt.clf()
    data_1.clear()
    data_2.clear()