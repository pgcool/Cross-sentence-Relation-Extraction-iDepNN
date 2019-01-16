# Computes macro averaged F1 score
# hypoCounts : # of predicted labels for a relations
# refCounts  : # of true labels for a relations
# posCounts  : # of true positives


ind2rel = {
    0 : "Localization",
    1 : "PartOf",
}


rel2ind = {
    "Localization" : 0,
    "PartOf" : 1,
}


def getIndex(relation):
    global rel2ind
    return rel2ind[relation]


relationList = ["Localization", "PartOf"]

def getRelation(index):
    global ind2rel
    return ind2rel[index]

def getMacroFScore(hypothesisList, referenceList):
    global relationList

    posCounts = {}
    hypoCounts = {}
    refCounts = {}
    precisions = {}
    recalls = {}
    fscores = {}

    for i in range(0,len(hypothesisList)):
        hypoInd = hypothesisList[i]
        refInd = referenceList[i]
        hypo = getRelation(hypoInd)
        ref = getRelation(refInd)
        # print(hypo, ref)
        if hypo == ref:
            if not hypo in posCounts:
                posCounts[hypo] = 0
            posCounts[hypo] += 1
        if not hypo in hypoCounts:
            hypoCounts[hypo] = 0
        hypoCounts[hypo] += 1
        if not ref in refCounts:
            refCounts[ref] = 0
        refCounts[ref] += 1
    relationsWithScores = list(set(list(hypoCounts.keys()) + list(refCounts.keys())))


    meanF = 0
    meanP = 0
    meanR = 0
    for rel in relationList:
        precision = 0
        recall = 0
        fscore = 0
        if rel in posCounts:
            if hypoCounts[rel] != 0:
                precision = posCounts[rel] * 1.0 / hypoCounts[rel]
            else:
                precision = 0.0
            if refCounts[rel] != 0:
                recall = posCounts[rel] * 1.0 / refCounts[rel]
            else:
                recall = 0.0
            if precision + recall != 0:
                fscore = 2 * precision * recall / (precision + recall)
            else:
                fscore = 0.0
            precisions[rel] = precision
            recalls[rel] = recall
            fscores[rel] = fscore
            meanF += fscore
            meanP += precision
            meanR += recall
    return meanF / len(relationList)
