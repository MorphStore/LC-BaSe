"""
The core of the cost model for lightweight integer compression algorithms.
"""

import lcbase_py.algo as algo

import pandas as pd

import math


#******************************************************************************
# Bit width of the uncompressed data
#******************************************************************************

UNCOMPR_BW = 32


#******************************************************************************
# Cost estimation
#******************************************************************************

COSTTYPE_PHY = 1
COSTTYPE_LOG_DI = 2
COSTTYPE_LOG_DD = 3
    
CONTEXT_STAND_ALONE = 1
CONTEXT_IN_CASC = 2

class CostModel:
    """
    A cost model for lightweight integer compression algorithms, including
    white-box and black-box part.
    """
    
    def __init__(self):
        # White-box part.
        
        self.costTypes = {}
        self.adaptFuncs = {}
        self.mixtureFuncs = {}
        self.changeFuncs = {}
        
        self.countValuesCalib = 0
        
        # Black-box part.
        
        self.diProfs = {CONTEXT_STAND_ALONE: {}, CONTEXT_IN_CASC: {}}
        self.ddProfs = {CONTEXT_STAND_ALONE: {}, CONTEXT_IN_CASC: {}}
        self.bwProfs = {CONTEXT_STAND_ALONE: {}, CONTEXT_IN_CASC: {}}
        
        self.penaltyFactors = {CONTEXT_STAND_ALONE: {}, CONTEXT_IN_CASC: {}}
        

    # Note that the objective is encoded as the mode of the Algo object.
    def cost(self, al, dfDC):
        if isinstance(al, algo.StandAloneAlgo):
            costType = self.costTypes[al.changeMode(None)]
            if costType == COSTTYPE_PHY:
                return self._costPhy(al, dfDC, CONTEXT_STAND_ALONE)
            elif costType == COSTTYPE_LOG_DI:
                return self._costLogDI(al, dfDC, CONTEXT_STAND_ALONE)
            elif costType == COSTTYPE_LOG_DD:
                return self._costLogDD(al, dfDC, CONTEXT_STAND_ALONE)
            else:
                raise RuntimeError("invalid cost type: '{}'".format(costType))
        elif isinstance(al, algo.CascadeAlgo):
            return self._costCasc(al, dfDC)
        else:
            raise RuntimeError("unsupported algorithm class")
    
    def _costPhy(self, al, dfDC, context):
        bwProfs = self.bwProfs[context]
        f = self._factor(al._mode, dfDC)
        sIsSorted = dfDC[ColsDC.isSorted]
        dfBwHistSorted = getBwHist(dfDC[sIsSorted])
        dfBwHistUnsorted = getBwHist(dfDC[~sIsSorted])
        # We calculate the cost separately for sorted and unsorted datasets.
        # In the end we reindex to restore the original order.
        return f * (
                # Costs for the cases where the data is sorted.
                dfBwHistSorted.dot(bwProfs[al])
                # Costs for the cases where the data is unsorted.
                .append(
                        self.adaptFuncs[al.changeMode(None)](dfBwHistUnsorted)
                        .dot(bwProfs[al]) + (
                            self._penalty(al, dfBwHistUnsorted, context)
                                if al._mode != algo.MODE_FORMAT
                                else 0
                        )
                )
        ).reindex(index=dfDC.index)

    def _costLogDI(self, al, dfDC, context):
        f = self._factor(al._mode, dfDC)
        return pd.Series(f * self.diProfs[context][al], index=dfDC.index)

    def _costLogDD(self, al, dfDC, context):
        res = []
        sDDProf = self.ddProfs[context][al]
        index = sDDProf.index
        isConstantEnd = True
        for idx, row in dfDC.iterrows():
            arg = row[index.name]
            if arg in index:
                # The argument is contained in the profile.
                # TODO why .loc[arg]? wouldn't [arg] suffice?
                estVal = sDDProf.loc[arg]
            elif isConstantEnd and arg > index[-1]:
                # The argument is not covered by the profile's domain, because
                # it is too large AND we assume that the values continue
                # CONSTANTLY after the domain covered by the profile.
                estVal = sDDProf.loc[index[-1]]
            else:
                # Linear interpolation of the value.
                if arg < index[0]:
                    # The argument is not covered by the profile's domain,
                    # because it is too small. Interpolation using the first
                    # two points.
                    prevPointArg = index[0]
                    nextPointArg = index[1]
                elif not isConstantEnd and arg > index[-1]:
                    # The argument is not covered by the profile's domain,
                    # because it is too large AND we assume that the values
                    # continue LINEARLY after the domain covered by the
                    # profile. Interpolation using the last two points.
                    prevPointArg = index[-2]
                    nextPointArg = index[-1]
                else:
                    # The argument is covered by the profile. Interpolation
                    # using the closest point before and after the argument.
                    prevPointArg = index[index.get_loc(arg, method="ffill")]
                    nextPointArg = index[index.get_loc(arg, method="bfill")]
                prevPointVal = sDDProf.loc[prevPointArg]
                nextPointVal = sDDProf.loc[nextPointArg]
                estVal = (nextPointVal - prevPointVal) / \
                    (nextPointArg - prevPointArg) * \
                    (arg - prevPointArg) + prevPointVal
            res.append(estVal)
        return self._factor(al._mode, dfDC) * pd.Series(res, index=dfDC.index)

    def _costCasc(self, al, dfDC):
        logAlgo = al.getLogAlgo()
        phyAlgo = al.getPhyAlgo()

        modePhy = algo.MODE_DECOMPR \
            if (al._mode == algo.MODE_AGG) \
            else al._mode
            
        # For the compression rate, there are no separate profiles for the use
        # of an algorithm in a cascade, so we employ the profiles for
        # stand-alone use.
        context = CONTEXT_STAND_ALONE \
            if al._mode == algo.MODE_FORMAT \
            else CONTEXT_IN_CASC
        
        costTypeLog = self.costTypes[logAlgo.changeMode(None)]
        if costTypeLog == COSTTYPE_LOG_DI:
            sCostLog = self._costLogDI(logAlgo, dfDC, context)
        elif costTypeLog == COSTTYPE_LOG_DD:
            sCostLog = self._costLogDD(logAlgo, dfDC, context)
        else:
            raise RuntimeError(
                    "invalid cost type for logical side of cascade: '{}'"
                    .format(costTypeLog)
            )
            
        costTypePhy = self.costTypes[phyAlgo.changeMode(None)]
        if costTypePhy != COSTTYPE_PHY:
            raise RuntimeError(
                    "invalid cost type for physical side of cascade: '{}'"
                    .format(costTypePhy)
            )
        sCostPhy = self._costPhy(
                phyAlgo.changeMode(modePhy),
                self._changeDC(logAlgo, dfDC),
                context
        )

        if al._mode == algo.MODE_FORMAT:
            return sCostLog * sCostPhy / UNCOMPR_BW
        elif al._mode in [algo.MODE_COMPR, algo.MODE_DECOMPR, algo.MODE_AGG]:
            return sCostLog + sCostPhy
        
    # This function assumes that the data is unsorted.
    def _penalty(self, al, dfBwHist, context):
        return self.mixtureFuncs[al.changeMode(None)](dfBwHist) * \
            self.penaltyFactors[context][al]

    def _changeDC(self, al, dfDC):
        outRows = []

        for idx, inRow in dfDC.iterrows():
            inCountValues = inRow[ColsDC.count]
            inIsSorted = inRow[ColsDC.isSorted]
            inBwHist = list(inRow[ColsDC.effBitsHist()] / inCountValues)
            inCountDistinct = inRow[ColsDC.distinct]
            inMin = inRow[ColsDC.min]
            inMax = inRow[ColsDC.max]

            outCountValues, outBwHist, outIsSorted = \
                self.changeFuncs[al.changeMode(None)](
                    inCountValues,
                    inIsSorted,
                    inBwHist,
                    inCountDistinct,
                    inMin,
                    inMax
                )
            outBwHist = list(map(lambda x: x * outCountValues, outBwHist))

            outRows.append([outCountValues, outIsSorted] + outBwHist)

        return pd.DataFrame(
            outRows,
            columns=[
                ColsDC.count,
                ColsDC.isSorted,
                *ColsDC.effBitsHist(),
            ],
            index=dfDC.index
        )
        
    def _factor(self, mode, dfDC):
        if mode == algo.MODE_FORMAT:
            return 1
        elif mode in [algo.MODE_COMPR, algo.MODE_DECOMPR, algo.MODE_AGG]:
            return dfDC[ColsDC.count] / self.countValuesCalib


#******************************************************************************
# Algorithm selection
#******************************************************************************

def select(
        algos, evalFunc, minimize, worstCaseComprRate, sUseThisLogAlgo=None
):
    # Just to find out the size and the index.
    sSomething = evalFunc(algos[0])

    sBestEval = pd.Series(
        [math.inf] * len(sSomething),
        index=sSomething.index
    )
    sBestAlgo = pd.Series(
        [None] * len(sSomething),
        index=sSomething.index
    )
    
    def getLogOrZero(al):
        if isinstance(al, algo.StandAloneAlgo):
            return 0 # None would be clearer, but causes problems later on.
        elif isinstance(al, algo.CascadeAlgo):
            return al.getLogAlgo()._name
        else:
            raise RuntimeError("unsupported type")
    
    if sUseThisLogAlgo is not None:
        sUseThisLogAlgo = sUseThisLogAlgo.apply(getLogOrZero)
        
    factor = 1 if minimize else -1

    for al in algos:
        sEval = factor * evalFunc(al)
        sAlgo = pd.Series([al] * len(sEval), index=sEval.index)

        sChoose = sEval < sBestEval
        
        if worstCaseComprRate != math.inf:
            sComprRate = evalFunc(al.changeMode(algo.MODE_FORMAT))
            sChoose &= (sComprRate < worstCaseComprRate)
        
        if sUseThisLogAlgo is not None:
            sChoose &= (sUseThisLogAlgo == getLogOrZero(al))
        
        sBestEval = sBestEval.mask(sChoose, sEval)
        sBestAlgo = sBestAlgo.mask(sChoose, sAlgo)

    return pd.DataFrame({"algo": sBestAlgo, "eval": factor * sBestEval})


#******************************************************************************
# Utilities
#******************************************************************************

class ColsDC:
    """
    Column names expected in the DataFrames containing data characteristics.
    """
    
    count = "countValues:"
    distinct = "countDistinctValues:"
    isSorted = "sortedAsc:"
    min = "minValue:"
    max = "maxValue:"
    
    _fsEffBitsHist = "effBitsHistogram[{}]:"
    
    @staticmethod
    def effBitsHist():
        return [
            ColsDC._fsEffBitsHist.format(bw) for bw in range(1, UNCOMPR_BW + 1)
        ]

def getBwHist(dfDC):
    dfBwHist = dfDC[ColsDC.effBitsHist()].div(dfDC[ColsDC.count], axis="index")
    dfBwHist.columns = range(1, UNCOMPR_BW + 1)
    return dfBwHist
