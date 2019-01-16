class ImageFeatures(object):

    t_dico = 1000

    def __init__(self, xs, ys, wordsim, id):
        self.id = id
        self.xs = xs
        self.ys = ys
        self.wordsim = wordsim

    def get_x(self):
        return self.xs

    def get_y(self):
        return self.ys

    def get_id(self):
        return self.id

    def get_words(self):
        return self.wordsim
