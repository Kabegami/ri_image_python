from path import Path
from imageNetParser import get_features
from index import compute_bow
import matplotlib.pyplot as plt

if __name__ == "__main__":
    DATA = "../data"
    directory = Path(DATA)
    features = []
    feat = get_features(DATA + "/taxi.txt")
    # for f in directory.files("*.txt"):
    #     feat = get_features(f)
    #     features.append(feat)

    bow = compute_bow(feat[0])
    plt.hist(bow)
    plt.show()
