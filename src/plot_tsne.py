
import numpy as np
from sklearn.manifold import TSNE
import argparse
import matplotlib.pyplot as plt
import os
import random
import matplotlib


def plot_tsne(root_dir, points_per_class, outfile)
    classNames = sorted(os.listdir(root_dir))
    features = []
    random.seed(42)
    for className in classNames:
        allFiles = sorted(os.listdir(os.path.join(root_dir, className)))
        usedFiles = random.sample(allFiles, points_per_class)
        for file in usedFiles:
            fp = os.path.join(root_dir, className, file)
            features.append(np.load(fp))
    features = np.stack(features)
    tsne = TSNE(n_components=2, verbose=1, perplexity=500, n_iter=5000)
    points = tsne.fit_transform(features)

    L=[[i]*points_per_class for i in range(len(classNames))]
    L=[item for color in L for item in color]
    colors = [plt.cm.tab10(i) for i in L] # using the "tab10" colormap
    handles = [matplotlib.patches.Patch(color=plt.cm.tab10(i), label=c) for i, c in enumerate(classNames)]

    plt.figure(figsize=(10,8))
    plt.scatter(*points.T, c=colors)
    plt.legend(handles=handles,  title='Color')
    plt.title("T-SNE plot with pretrained ResNet-18")
    plt.savefig(outfile)


def main():
    parser = argparse.ArgumentParser(description='Process args for t-SNE plot')
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--points_per_class", type=int, default=100)
    parser.add_argument("--outfile", type=str, required=True)
    args = parser.parse_args()
    print(args,flush=True)
    plot_tsne(args.root_dir, args.points_per_class, args.outfile)


if __name__=="__main__":
    main()
