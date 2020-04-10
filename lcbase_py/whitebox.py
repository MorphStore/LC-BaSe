"""
A collection of useful functions for configuring the white-box part of the cost
model for lightweight integer compression algorithms.
"""

import lcbase_py.costmodel as cm

import numpy as np
import pandas as pd

import math


# *****************************************************************************
# Adapt-functions
# *****************************************************************************
# Adapt the bit width histogram of the data to the algorithm.

def adaptId(dfBwHist):
    return dfBwHist

def adaptFixed(dfBwHist, blockSize):
    dfTriU = pd.DataFrame(np.triu([1.0] * cm.UNCOMPR_BW))
    dfTriU2 = dfTriU - pd.DataFrame(np.identity(cm.UNCOMPR_BW))
    dfTriU.columns = range(1, cm.UNCOMPR_BW + 1)
    dfTriU.index = range(1, cm.UNCOMPR_BW + 1)
    dfTriU2.columns = range(1, cm.UNCOMPR_BW + 1)
    dfTriU2.index = range(1, cm.UNCOMPR_BW + 1)
    return dfBwHist.dot(dfTriU) ** blockSize - \
        dfBwHist.dot(dfTriU2) ** blockSize

def adaptMax(dfBwHist, granularity="bit"):
    newData = []
    for idx, row in dfBwHist.iterrows():
        for bw in reversed(range(1, cm.UNCOMPR_BW + 1)):
            if row[bw] > 0:
                maxBw = bw
                break
        if granularity == "bit":
            pass
        elif granularity == "even":
            maxBw = maxBw + maxBw % 2
        elif granularity == "byte":
            maxBw = maxBw + (8 - maxBw % 8)
        elif granularity == "pot": # power of two
            maxBw = 2 ** (maxBw - 1).bit_length() # only for maxBw >= 1
        else:
            raise RuntimeError("unsupported granularity")
        newRow = [0] * cm.UNCOMPR_BW
        newRow[maxBw - 1] = 1
        newData.append(newRow)
    dfNew = pd.DataFrame(
            newData, columns=range(1, cm.UNCOMPR_BW + 1), index=dfBwHist.index
    )
    return dfNew

def adaptFixedPFOR(dfBwHist, blockSize, bestInternalCost):
    resRowList = []
    for idx, row in dfBwHist.iterrows():
        # We do not use the bit width that SIMD-FastPFOR chooses (which is
        # bestBw), but instead we use an imaginary bit width that also accounts
        # for the cost of the exceptions.
        # TODO Smoothen this somehow by not setting a single bit width to 1,
        #      but by setting multiple (at least two) bit widths; this could
        #      avoid the resulting step shape of the compression rate.
        newRow = [0] * 32
        newRow[int(round(bestInternalCost(blockSize, row) / blockSize)) - 1] = 1
        resRowList.append(newRow)
    dfRes = pd.DataFrame(
            resRowList, index=dfBwHist.index, columns=range(1, 32 + 1)
    )
    return dfRes

def adaptVar(dfBwHist, partitioningRoutine):
    newRows = []
    for rowIdx, row in dfBwHist.iterrows():
        # TODO Do not hardcode this number.
        countElemsTotal = 10000
        bws = np.random.choice(
                list(range(1, cm.UNCOMPR_BW + 1)), countElemsTotal, p=list(row)
        )
        
        partInfo = partitioningRoutine(bws)
        
        # This number should be the same as countElemsTotal. However, since the
        # partitioning routine might exclude the last, possibly incomplete
        # block (for simplicity), this number could be lower.
        countElemsInPartitions = 0
        for x in partInfo:
            x[_ADAPTVAR_COUNT] *= x[_ADAPTVAR_BS]
            countElemsInPartitions += x[_ADAPTVAR_COUNT]
        newRow = [0] * cm.UNCOMPR_BW
        for blockSize, bitWidth, countElems in partInfo:
            newRow[bitWidth - 1] = countElems / countElemsInPartitions
        newRows.append(newRow)
    dfRes = pd.DataFrame(
            newRows, index=dfBwHist.index, columns=range(1, cm.UNCOMPR_BW + 1)
    )
    
    return dfRes


# *****************************************************************************
# Mixture-functions
# *****************************************************************************

def mixtureZero(dfBwHist):
    return 0

def mixtureSIMDFastPFor(dfBwHist):
    sharesExcept = []
    for idx, rowBwHist in dfBwHist.iterrows():
        sharesExcept.append(_internal_SimdFastPFor(128, rowBwHist)[1])
    return pd.Series(sharesExcept, index=dfBwHist.index)

_dfTo7BitUnitHist = pd.DataFrame(
    [[1, 0, 0, 0, 0]] * 7 + \
    [[0, 1, 0, 0, 0]] * 7 + \
    [[0, 0, 1, 0, 0]] * 7 + \
    [[0, 0, 0, 1, 0]] * 7 + \
    [[0, 0, 0, 0, 1]] * 4,
    index=range(1, 32 + 1)
)

def mixtureMaskedVByte(dfBwHist):
    df7BitUnitHist = dfBwHist.dot(_dfTo7BitUnitHist)
    sMaxRelFreq = df7BitUnitHist.max(axis=1)
    return sMaxRelFreq.mask(sMaxRelFreq > 0.5, 1 - sMaxRelFreq)


# *****************************************************************************
# Change-functions
# *****************************************************************************

def _bitwidth(n):
    # This formula does not work for too large numbers (of more then 48 bits or
    # so), because, due to the limited precision, log2(n+1) == log2(n), then.
    # return 1 if n == 0 else int(math.ceil(math.log2(n + 1)))
    return 1 if n == 0 else int(n).bit_length()

def _minByBitwidth(bw):
    return 0 if bw == 1 else 2 ** (bw - 1)
    
def _maxByBitwidth(bw):
    return 1 if bw == 1 else (2 ** bw) - 1

def _overlap(min1, max1, min2, max2):
    if max1 < min2 or max2 < min1:
        return 0
    return min(max1, max2) - max(min1, min2) + 1

def changeRle(
        inCountValues, inIsSorted, inBwHist, inCountDistinct, inMin, inMax
):
    outBwHist = list(map(lambda x: x / 2, inBwHist))
    outIsSorted = False
    if inIsSorted:
        outCountValues = 2 * inCountDistinct
        inAvgRl = inCountValues / inCountDistinct
        outBwHist[_bitwidth(inAvgRl) - 1] += 0.5
    else:
        outCountValues = 2 * inCountValues
        outBwHist[_bitwidth(1) - 1] += 0.5
    return outCountValues, outBwHist, outIsSorted

def changeDelta(
        inCountValues, inIsSorted, inBwHist, inCountDistinct, inMin, inMax
):
    outCountValues = inCountValues
    outIsSorted = False
    outBwHist = [0] * cm.UNCOMPR_BW
    if inIsSorted:
        vs = 4 # TODO Do not hard-code the vector size.
        inCountValues2 = inCountValues / vs
        inCountDistinct2 = min(inCountDistinct, inCountValues2)
        
        inCountBreaks = inCountDistinct2 - 1
        inLowestNonEmptyBw = _bitwidth(inMin)
        outBwHist[inLowestNonEmptyBw - 1] += 1
        outBwHist[_bitwidth(0) - 1] += inCountValues2 - 1 - inCountBreaks
        if inCountBreaks > 0:
            inAvgBreakDelta = (inMax - inMin) / inCountBreaks
            outBwHist[_bitwidth(inAvgBreakDelta) - 1] += inCountBreaks
        outBwHist = list(map(lambda x: x / inCountValues2, outBwHist))
    else:
        inAvgAbsDelta = (inMax - inMin) / 3
        outBwHist[_bitwidth(inAvgAbsDelta) - 1] += 0.5
        outBwHist[_bitwidth(2 ** cm.UNCOMPR_BW - 1 - inAvgAbsDelta) - 1] += 0.5
    return outCountValues, outBwHist, outIsSorted

def changeFor(
        inCountValues, inIsSorted, inBwHist, inCountDistinct, inMin, inMax
):
    outCountValues = inCountValues
    outIsSorted = inIsSorted
    outBwHist = [0] * cm.UNCOMPR_BW
    inLowestNonEmptyBw = _bitwidth(inMin)
    inHighestNonEmptyBw = _bitwidth(inMax)
    for inBw in range(1, cm.UNCOMPR_BW + 1):
        if inBwHist[inBw - 1] > 0:
            inBucketMinShifted = (
                inMin if inBw == inLowestNonEmptyBw  else _minByBitwidth(inBw)
            ) - inMin
            inBucketMaxShifted = (
                inMax if inBw == inHighestNonEmptyBw else _maxByBitwidth(inBw)
            ) - inMin
            inBucketSize = inBucketMaxShifted - inBucketMinShifted + 1
            for outBw in range(
                _bitwidth(inBucketMinShifted),
                _bitwidth(inBucketMaxShifted) + 1
            ):
                outBucketMin = _minByBitwidth(outBw)
                outBucketMax = _maxByBitwidth(outBw)
                outBwHist[outBw - 1] += _overlap(
                        inBucketMinShifted, inBucketMaxShifted,
                        outBucketMin, outBucketMax
                ) / inBucketSize * inBwHist[inBw - 1]
    return outCountValues, outBwHist, outIsSorted

def changeDict(
        inCountValues, inIsSorted, inBwHist, inCountDistinct, inMin, inMax
):
    outCountValues = inCountValues
    outIsSorted = inIsSorted
    outBwHist = [0] * cm.UNCOMPR_BW
    prevInBucketMaxKey = -1
    for inBw in range(1, cm.UNCOMPR_BW + 1):
        inCountDistinctBw = round(inBwHist[inBw - 1] * inCountDistinct)
        if inCountDistinctBw > 0:
            inBucketMinKey = prevInBucketMaxKey + 1
            inBucketMaxKey = inBucketMinKey + inCountDistinctBw - 1
            prevInBucketMaxKey = inBucketMaxKey
            for outBw in range(
                _bitwidth(inBucketMinKey), _bitwidth(inBucketMaxKey) + 1
            ):
                outBucketMin = _minByBitwidth(outBw)
                outBucketMax = _maxByBitwidth(outBw)
                outBwHist[outBw - 1] += _overlap(
                        inBucketMinKey, inBucketMaxKey,
                        outBucketMin, outBucketMax
                ) / inCountDistinctBw * inBwHist[inBw - 1]
    # The problem is that (due to rounding errors?) the sum over the bitwidth
    # histogram is sometimes slightly less than 1.
    # TODO This solution might increase the probability of bit width 32 from
    #      zero to non-zero; maybe the probability of the smallest or largest
    #      bit width already having a non-zero probability should be reset.
    outBwHist[cm.UNCOMPR_BW - 1] = 1 - sum(outBwHist[:cm.UNCOMPR_BW - 1])
    return outCountValues, outBwHist, outIsSorted


# *****************************************************************************
# Internal
# *****************************************************************************

def _internal_SimdFastPFor(blockSize, rowBwHist):
    overheadPerException = 8
    bestBw = 32 # the bit width yielding the best total cost known so far
    while rowBwHist[bestBw] == 0:
        bestBw -= 1
    maxBw = bestBw # the maximum bit width occuring in the data
    bestCost = bestBw * blockSize # the best total cost known so far
    shareExcept = 0
    # the share of exceptions implied by the bit width yielding the best
    # total cost known so far
    bestShareExcept = shareExcept
    for bw in reversed(range(1, bestBw - 1 + 1)):
        shareExcept += rowBwHist[bw + 1]
        thisCost = blockSize * shareExcept * overheadPerException + \
            blockSize * shareExcept * (maxBw - bw) + bw * blockSize + 8
        if thisCost < bestCost:
            bestCost = thisCost
            bestBw = bw
            bestShareExcept = shareExcept
    return bestCost, bestShareExcept

def _best_internalCost_SimdFastPFor(blockSize, rowBwHist):
    return _internal_SimdFastPFor(blockSize, rowBwHist)[0]

_ADAPTVAR_BS = 0 # block size
_ADAPTVAR_BW = 1 # bit width
_ADAPTVAR_COUNT = 2

_modiSimdGroupSimple128 = [
    # (block size, bit width)
    (4 * 32,  1),
    (4 * 16,  2),
    (4 * 10,  3),
    (4 *  8,  4),
    (4 *  6,  5),
    (4 *  5,  6),
    (4 *  4,  8),
    (4 *  3, 10),
    (4 *  2, 16),
    (4 *  1, 32),
]

def _partitioningRoutine_SimdGroupSimple(bws, modi):
    res = [[blockSize, bitWidth, 0] for blockSize, bitWidth in modi]
    while len(bws) > 0:
        for modeIdx, (blockSize, bitWidth) in enumerate(modi):
            pos = 0
            while pos < blockSize and pos < len(bws) and bws[pos] <= bitWidth:
                pos += 1
            if pos == blockSize:
                res[modeIdx][_ADAPTVAR_COUNT] += 1
                bws = bws[blockSize:]
                break
            elif pos == len(bws):
                # End reached, we ignore the last, incomplete block.
                bws = []
                break
    return res