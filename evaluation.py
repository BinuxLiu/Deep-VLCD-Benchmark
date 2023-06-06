import os

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import numpy as np
import matplotlib.pyplot as plt

methods = ["sift", "disk", "loftr", "superglue", "gms"]

# methods = ["sift_a", "sift_b", "superglue_a", "superglue_b"]

# methods = ["superglue_a", "superglue_197_20"]
methods = ["superglue_197_20_DTC"]

gt_files = [file for file in os.listdir(f"./output/{methods[0]}/") if file.endswith(".txt")]

for gt_file in gt_files:
    dataset_name = gt_file.strip(".txt")
    print(dataset_name)
    print("--------------------------------------------------------------")
    dataset_name = "robotcar_qAutumn_dbSuncloud_easy_final"
    for method in methods:
        print(method)
        print("--------------------------------------------------------------")
        match_points = []
        labels = []
        with open(f"./output/{method}/{dataset_name}.txt", 'r') as input:
            
            lines = input.readlines()
            lcd_results = [tuple(line.strip().split(', ')) for line in lines]
            for query_img_path, ref_img_path, gt, n_matches in lcd_results:
                match_points.append(int(n_matches))
                labels.append(int(gt))
            match_points = np.array(match_points)
            labels = np.array(labels)
            scaled_scores = match_points / max(match_points)
            precision, recall, threshold = precision_recall_curve(labels, scaled_scores)
            idx = np.where(precision == 1.0)[0]
            print("Maximum recall at 100% precision:", recall[idx[0]])
            if recall[idx[0]] > 0:
                print("Points Threshold with Maximum recall at 100% precision:", threshold[idx[0]] * max(match_points))
            else: 
                print("NA")
            average_precision = average_precision_score(labels, scaled_scores)
            plt.plot(recall, precision, label="{} (AP={:.3f})".format(method, average_precision))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc = 'lower right')
    plt.title(f"Precision-Recall Curves on {dataset_name}")
    plt.savefig("./output/plot/pr_curve_{}.png".format(dataset_name))
    plt.close()
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
    print("--------------------------------------------------------------")
            



    
