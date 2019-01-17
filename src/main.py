#from path import Path
from imageNetParser import get_features, get_classes_image_net
from index import compute_bow, file_to_dataset, get_dataset, save, load, plot_histo, sample
import matplotlib.pyplot as plt
import model
from sklearn.decomposition import PCA
from tools import accuracy

if __name__ == "__main__":
    DATA = "../data"
    #directory = Path(DATA)
    features = []
    fname = DATA + "/taxi.txt"
    
    feat = get_features(DATA + "/taxi.txt")
    # for f in directory.files("*.txt"):
    #     feat = get_features(f)
    #     features.append(feat)
    
    #bow = compute_bow(feat[0])
    # print("bow : ", bow)
    #plot_histo(bow)
    #dataset = get_dataset(DATA, get_classes_image_net())
    #save(dataset)
    #dataset = load()
    dimpsi = 250 * 9
    
    x,y = sample(dataset, 1, train=False)
    y = y[0]
    print(x.shape, y.shape)
    
    linear = model.LinearStructModel_Ex(dimpsi)
    linear.instantiation()
    print(linear.predict(x))
    print(linear.lai(x,y))
    
    nbit=100
    classifier = model.GenericTrainingAlgorithm(dimpsi)
    print(accuracy(classifier, dataset))
    L = classifier.fit(dataset, nb_it=nbit, register=True, alpha=1e-6, lr=1e-2)
    train_acc, test_acc = zip(*L)
    print(accuracy(classifier, dataset))
    plt.plot(list(range(nbit)), train_acc, label="train")
    plt.plot(list(range(nbit)), test_acc, label="test")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
    
    from sklearn.metrics import confusion_matrix
    pred = [classifier.predict(x) for x in dataset.x_test]
    mat = confusion_matrix(dataset.y_test, pred)
    
    #save(dataset)
    # dataset = load()
    # print([x.shape for x in dataset])
    
    
plt.imshow(mat)
plt.xticks(range(9), classes, rotation=90)
plt.yticks(range(9), classes)
plt.colorbar()
plt.show()


