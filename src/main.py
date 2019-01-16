from path import Path
from imageNetParser import get_features, get_classes_image_net
from index import compute_bow, file_to_dataset, get_dataset, save, load
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

if __name__ == "__main__":
    DATA = "../data"
    directory = Path(DATA)
    features = []
    fname = DATA + "/taxi.txt"
    
    feat = get_features(DATA + "/taxi.txt")
    # for f in directory.files("*.txt"):
    #     feat = get_features(f)
    #     features.append(feat)
    
    bow = compute_bow(feat[0])
    # print("bow : ", bow)

    dateset = get_dataset(DATA, get_classes_image_net())
    save(dataset)
    # dataset = load()
    # print([x.shape for x in dataset])
    
    
