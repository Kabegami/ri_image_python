#!/usr/bin/env python
""" generated source for module ranking """
import java.util.ArrayList

import java.util.Collections

import java.util.List

import java.util.Random

import upmc.ri.struct.DataSet

import upmc.ri.struct.STrainingSample

import upmc.ri.struct.ranking.RankingOutput

import upmc.ri.utils.Pair

import upmc.ri.utils.VectorOperations

class RankingFunctions(object):
    """ generated source for class RankingFunctions """
    @classmethod
    def recalPrecisionCurve(cls, y):
        """ generated source for method recalPrecisionCurve """
        precision = ArrayList()
        recall = ArrayList()
        nbPlus = y.getNbPlus()
        top = 0
        i = 0
        while i < y.getRanking().size():
            #  Computing recall for the ist example : R(i) = #  relevant docs at i / # relevant docs
            #  Computing precision for the ist example : R(i) = #  relevant docs at i / i
            if y.getLabelsGT().get(y.getRanking().get(i)) == 1:
                #  Check is the top ist elet is positive
                top += 1
            precision.add(top / float((i + 1)))
            recall.add(top / float(nbPlus))
            i += 1
        #  Computing recall/precision curve
        rp = [None]*2
        rp[0][0] = 0.0
        rp[1][0] = 1.0
        j = 1
        while j <= len(recall):
            rp[0][j] = recall.get(j - 1)
            j += 1
        j = 1
        while j <= len(recall):
            rp[1][j] = precision.get(j - 1)
            j += 1
        return rp

    @classmethod
    def averagePrecision(cls, y):
        """ generated source for method averagePrecision """
        rp = cls.recalPrecisionCurve(y)
        AP = 0.0
        j = 0
        while j < len(length):
            AP += (rp[1][j + 1] + rp[1][j]) * (rp[0][j + 1] - rp[0][j]) / 2.0
            j += 1
        return AP

    @classmethod
    def loss_augmented_inference(cls, ts, w):
        """ generated source for method loss_augmented_inference """
        sortedPlus = ArrayList()
        sortedMinus = ArrayList()
        nbPlus = ts.output.getNbPlus()
        nbMinus = len(ts.input) - nbPlus
        pairsPlus = ArrayList()
        i = 0
        while i < len(ts.input):
            if ts.output.getLabelsGT().get(i) == 1:
                pairsPlus.add(Pair(i, VectorOperations.dot(w, ts.input.get(i))))
            i += 1
        Collections.sort(pairsPlus, Collections.reverseOrder())
        i = 0
        while i < len(pairsPlus):
            sortedPlus.add(pairsPlus.get(i).getKey())
            i += 1
        pairsMinus = ArrayList()
        i = 0
        while i < len(ts.input):
            if ts.output.getLabelsGT().get(i) == -1:
                pairsMinus.add(Pair(i, VectorOperations.dot(w, ts.input.get(i))))
            i += 1
        Collections.sort(pairsMinus, Collections.reverseOrder())
        i = 0
        while i < len(pairsMinus):
            sortedMinus.add(pairsMinus.get(i).getKey())
            i += 1
        imaxs = ArrayList(Collections.nCopies(nbMinus, 0))
        j = 0
        while j < nbMinus:
            while k < nbPlus:
                deltasij.add(deltaij)
                k += 1
            while k < nbPlus:
                while h < nbPlus:
                    val += deltasij.get(h)
                    h += 1
                if val > valmax:
                    valmax = val
                    imax = k
                k += 1
            imaxs.set(j, imax)
            j += 1
        res = fusionList(sortedPlus, sortedMinus, imaxs)
        qo = RankingOutput(nbPlus, res, ts.output.getLabelsGT())
        return qo

    @classmethod
    def val_optj(cls, j, k, skp, sjn, nbPlus, nbMinus):
        """ generated source for method val_optj """
        jj = j + 1
        kk = k + 1
        val = 1 / nbPlus * (jj / (jj + kk) - (jj - 1) / (jj + kk - 1)) - 2.0 * (skp - sjn) / (nbPlus * nbMinus)
        return val

    @classmethod
    def fusionList(cls, l1, l2, pos):
        """ generated source for method fusionList """
        if len(l2) != len(pos):
            System.err.println(" Error fusionList ! l2 must be the same size than pos !")
            return None
        res = ArrayList(l1)
        i = 0
        while i < len(l2):
            while j < i:
                if pos.get(j) < pos.get(i):
                    dec += 1
                j += 1
            res.add(pos.get(i) + dec, l2.get(i))
            i += 1
        return res

    @classmethod
    def convertClassif2Ranking(cls, data, classquery, vectors, ranking_id):
        """ generated source for method convertClassif2Ranking """
        vectors = []
        ranking_id = int()
        super(RankingFunctions, self).__init__()
        self.vectors = vectors
        self.ranking_id = ranking_id
        res = None
        listtrain = ArrayList()
        ltrain = ArrayList()
        rankingtrain = ArrayList()
        outputtrain = None
        listtmp = ArrayList()
        nbPlus = 0
        nbMinus = 0
        for ts in data.listtrain:
            if ts.output == classquery:
                listtmp.add(RankingData(ts.input, nbPlus))
                nbPlus += 1
        for ts in data.listtrain:
            if not ts.output == classquery:
                listtmp.add(RankingData(ts.input, nbPlus + nbMinus))
                nbMinus += 1
        Collections.shuffle(listtmp, Random(1000))
        i = 0
        while i < len(listtmp):
            ltrain.add(listtmp.get(i).vectors)
            rankingtrain.add(listtmp.get(i).ranking_id)
            i += 1
        rankingtrain = swapRankingPositionning(rankingtrain)
        labelsGTtrain = labelsfromrank(rankingtrain, nbPlus)
        outputtrain = RankingOutput(nbPlus, rankingtrain, labelsGTtrain)
        listtrain.add(STrainingSample(ltrain, outputtrain))
        print "************ classinput=" + classquery + " ltrain=" + len(ltrain) + " rankingtrain=" + len(rankingtrain) + " ************"
        listtest = ArrayList()
        ltest = ArrayList()
        rankingtest = ArrayList()
        outputtest = None
        nbPlusTest = 0
        nbMinusTest = 0
        for ts in data.listtest:
            if ts.output == classquery:
                ltest.add(ts.input)
                rankingtest.add(nbPlusTest)
                nbPlusTest += 1
        for ts in data.listtest:
            if not ts.output == classquery:
                ltest.add(ts.input)
                rankingtest.add(nbPlusTest + nbMinusTest)
                nbMinusTest += 1
        rankingtest = swapRankingPositionning(rankingtest)
        labelsGTtest = labelsfromrank(rankingtest, nbPlusTest)
        outputtest = RankingOutput(nbPlusTest, rankingtest, labelsGTtest)
        listtest.add(STrainingSample(ltest, outputtest))
        print "************ nbPlus train=" + nbPlus + " nbMinus train=" + nbMinus + " nbPlus test=" + nbPlusTest + " nbMinus test=" + nbMinusTest + " ************"
        res = DataSet(listtrain, listtest)
        return res

    @classmethod
    def swapRankingPositionning(cls, input):
        """ generated source for method swapRankingPositionning """
        output = ArrayList(input)
        i = 0
        while i < len(input):
            output.set(input.get(i), i)
            i += 1
        return output

    @classmethod
    def labelsfromrank(cls, ranking, nbPlus):
        """ generated source for method labelsfromrank """
        labels = ArrayList()
        i = 0
        while i < len(ranking):
            labels.add(-1)
            i += 1
        i = 0
        while i < nbPlus:
            labels.set(ranking.get(i), 1)
            i += 1
        return labels

    @classmethod
    def labelsfrompositionning(cls, pos, nbPlus):
        """ generated source for method labelsfrompositionning """
        labels = ArrayList()
        i = 0
        while i < len(pos):
            labels.add(-1)
            i += 1
        i = 0
        while i < len(pos):
            if pos.get(i) < nbPlus:
                labels.set(i, 1)
            i += 1
        return labels

