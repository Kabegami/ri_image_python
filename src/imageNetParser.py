from src.index.imageFeatures import ImageFeatures


class ImageNetParser:

    @staticmethod
    def get_words(filename):
        res = []

        file = open(filename, 'r')
        file.readline()

        for _ in file:
            line = file.readline()[1:-2]

            # 3 lines for skipping x-y-BB
            file.readline()
            file.readline()
            file.readline()

            res.append([int(w) for w in line.split(";", 1100)[:-1]])
        file.close()

        return res

    @staticmethod
    def get_features(filename):
        res = []

        file = open(filename, 'r')
        file.readline()

        while True:
            # reading id
            id = file.readline()
            if not id:
                break

            # reading words
            line = file.readline()[1:-2]  # remove '[' and ']'
            wordsim = [int(w) for w in line.split(";", 1100)[:-1]]

            # reading x
            line = file.readline()[1:-2]  # remove '[' and ']'
            xs = [float(w) for w in line.split(";", 1100)[:-1]]

            # reading y
            line = file.readline()[1:-2]  # remove '[' and ']'
            ys = [float(w) for w in line.split(";", 1100)[:-1]]

            res.append(ImageFeatures(xs, ys, wordsim, id))
            file.readline()

        file.close()

        return res

    classes_image_net = ["taxi",
                         "ambulance",
                         "minivan",
                         "acoustic_guitar",
                         "electric_guitar",
                         "harp",
                         "wood-frog",
                         "tree-frog",
                         "european_fire_salamander"]
