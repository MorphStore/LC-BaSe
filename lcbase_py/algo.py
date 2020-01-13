"""
Internal representations for lightweight integer compression algorithms and
cascades thereof.
"""

import lcbase_py.utils as utils

import re


MODE_FORMAT  = 1
MODE_COMPR   = 2
MODE_DECOMPR = 3
MODE_AGG     = 4

# These suffixes should be kept consistent with those used in the C++ core of
# the benchmark framework.
_suffixesByMode = {
    MODE_FORMAT : None,
    MODE_COMPR  : "c",
    MODE_DECOMPR: "d",
    MODE_AGG    : "Sum",
}
_modesBySuffix = {s: m for m, s in _suffixesByMode.items()}

_suffixUnkownMode = "?"

displayNames = dict()
specialFormatNames = dict()

_allAlgoModeSuffixes = [
    s for m, s in _suffixesByMode.items() if m is not MODE_FORMAT
]

class Algo:
    def __init__(self, parsedFromFormatName):
        self._parsedFromFormatName = parsedFromFormatName
        
    def __str__(self):
        return "<{}:{}>".format(
                self.__class__.__name__, self.getInternalName()
        )
    
    def __repr__(self):
        return self.__str__()
    
    def __gt__(self, other):
        return self.__hash__() > other.__hash__()
    
    def _checkChangeMode(self):
        # TODO Actually, raising an exception on changeMode() is only necessary
        # if the format name cannot be mapped to an algorithm name without
        # ambiguities, i.e., often it would be ok.
        if self._parsedFromFormatName:
            raise RuntimeError(
                    "this instance of {} was parsed from a format name, thus, "
                    "its mode should not be changed".format(
                            self.__class__.__name__
                    )
            )

class StandAloneAlgo(Algo):
    _pAlgo = re.compile("(.+)_({})".format("|".join(_allAlgoModeSuffixes)))

    def __init__(self, name, mode=None, parsedFromFormatName=False):
        super().__init__(parsedFromFormatName)
        self._name = name
        self._mode = mode
        
    def __hash__(self):
        # TODO Non-dummy implementation.
        return 0
        
    def __eq__(self, other):
        if isinstance(other, StandAloneAlgo):
            # The idea is that two StandAloneAlgos are equal if their modes and
            # their names are equal. However, one of the StandAloneAlgos may
            # have been parsed from a format name. If this format is shared by
            # different algorithms, we cannot directly compare the names.
            # Instead, we must map them again through the special format names
            # dictionary.
            return (
                other._mode == self._mode and
                (
                    (
                        self._mode == MODE_FORMAT and
                        specialFormatNames.get(other._name, other._name) == \
                            specialFormatNames.get(self._name, self._name)
                    ) or (
                        self._mode != MODE_FORMAT and
                        other._name == self._name
                    )
                )
            )
        elif isinstance(other, Algo):
            return False
        else:
            return NotImplemented
        
    @staticmethod
    def fromAlgoName(algoName):
        m = StandAloneAlgo._pAlgo.fullmatch(algoName)
        if m is None:
            return None
        else:
            name = m.group(1)
            mode = _modesBySuffix[m.group(2)]
            return StandAloneAlgo(name, mode)
        
    @staticmethod
    def fromFormatName(formatName):
        specialFormatNamesInv = {v: k for k, v in specialFormatNames.items()}
        return StandAloneAlgo(
                specialFormatNamesInv.get(formatName, formatName),
                MODE_FORMAT,
                True
        )
    
    def getInternalName(self):
        if self._mode == MODE_FORMAT:
            return specialFormatNames.get(self._name, self._name)
        elif self._mode is not None:
            return "{}_{}".format(self._name, _suffixesByMode[self._mode])
        else:
            return "{}_{}".format(self._name, _suffixUnkownMode)
    
    def getDisplayName(self, extraDisplayNames=None):
        return utils.getFromDicts(extraDisplayNames, displayNames, self._name)
            
    def changeMode(self, mode):
        self._checkChangeMode()
        return StandAloneAlgo(self._name, mode)

class CascadeAlgo(Algo):
    # The prefixes "Transformator" and "Format" can optionally precede the word
    # "Cascade", because the names used to be like that in previous versions
    # of the benchmark framework.
    _pAlgo = re.compile(
            "(?:Transformator)?Cascade\\((\d+):(.+)_({});(.+)_({})\\)".format(
                "|".join([
                    _suffixesByMode[MODE_COMPR],
                    _suffixesByMode[MODE_DECOMPR]
                ]),
                "|".join(_allAlgoModeSuffixes)
            )
    )
    _pFormat = re.compile("(?:Format)?Cascade\\((\d+):(.+);(.+)\\)")
    
    _fsInternalNameFormat = "Cascade({bs}:{log};{phy})"
    _fsInternalNameTransf = \
        "Cascade({bs}:{first}_{firstmode};{second}_{secondmode})"
    
    def __init__(self, log, phy, bs, mode=None, parsedFromFormatName=False):
        super().__init__(parsedFromFormatName)
        
        for attrName, paramVal in [("_log", log), ("_phy", phy)]:
            if isinstance(paramVal, str):
                setattr(self, attrName, paramVal)
            elif isinstance(paramVal, StandAloneAlgo):
                setattr(self, attrName, paramVal._name)
            else:
                raise RuntimeError(
                        "parameter {} must be either a str or a {}".format(
                                attrName[1:], StandAloneAlgo.__class__.__name__
                        )
                )
        
        self._bs = bs
        self._mode = mode
        
    def __hash__(self):
        # TODO Non-dummy implementation.
        return 0
        
    def __eq__(self, other):
        if isinstance(other, CascadeAlgo):
            # See the remark in StandAloneAlgo.__eq__, the same idea applies
            # here.
            return (
                other._mode == self._mode and
                other._bs == self._bs and
                (
                    (
                        self._mode == MODE_FORMAT and
                        specialFormatNames.get(other._log, other._log) == \
                            specialFormatNames.get(self._log, self._log) and
                        specialFormatNames.get(other._phy, other._phy) == \
                            specialFormatNames.get(self._phy, self._phy)
                    ) or (
                        self._mode != MODE_FORMAT and
                        other._log == self._log and
                        other._phy == self._phy
                    )
                )
            )
        elif isinstance(other, Algo):
            return False
        else:
            return NotImplemented
        
    @staticmethod
    def fromAlgoName(algoName):
        m = CascadeAlgo._pAlgo.fullmatch(algoName)
        if m is None:
            return None
        else:
            first = m.group(2)
            second = m.group(4)
            bs = int(m.group(1))
            mode = _modesBySuffix[m.group(5)]
            if mode == MODE_COMPR:
                log = first
                phy = second
            else:
                log = second
                phy = first
            # TODO Validate the combination.
            return CascadeAlgo(log, phy, bs, mode)
        
    @staticmethod
    def fromFormatName(formatName):
        specialFormatNamesInv = {v: k for k, v in specialFormatNames.items()}
        m = CascadeAlgo._pFormat.fullmatch(formatName)
        if m is None:
            return None
        else:
            log = m.group(2)
            phy = m.group(3)
            bs = int(m.group(1))
            # TODO Validate the combination.
            return CascadeAlgo(
                specialFormatNamesInv.get(log, log),
                specialFormatNamesInv.get(phy, phy),
                bs, MODE_FORMAT, True
            )
    
    def getInternalName(self):
        if self._mode == MODE_FORMAT:
            return CascadeAlgo._fsInternalNameFormat.format(
                bs=self._bs,
                log=specialFormatNames.get(self._log, self._log),
                phy=specialFormatNames.get(self._phy, self._phy)
            )
        else:
            if self._mode == MODE_COMPR:
                first = self._log
                second = self._phy
                modeSuffixFirst  = _suffixesByMode[MODE_COMPR]
                modeSuffixSecond = _suffixesByMode[MODE_COMPR]
            elif self._mode is not None:
                first = self._phy
                second = self._log
                modeSuffixFirst  = _suffixesByMode[MODE_DECOMPR]
                modeSuffixSecond = _suffixesByMode[self._mode]
            else:
                first = self._phy
                second = self._log
                modeSuffixFirst = _suffixUnkownMode
                modeSuffixSecond = _suffixUnkownMode
            return CascadeAlgo._fsInternalNameTransf.format(
                bs=self._bs,
                first=first, firstmode=modeSuffixFirst,
                second=second, secondmode=modeSuffixSecond
            )
    
    def getDisplayName(self, extraDisplayNames=None):
        return "{} + {}".format(
            utils.getFromDicts(extraDisplayNames, displayNames, self._log),
            utils.getFromDicts(extraDisplayNames, displayNames, self._phy)
        )
            
    def changeMode(self, mode):
        self._checkChangeMode()
        return CascadeAlgo(self._log, self._phy, self._bs, mode)
    
    def getLogAlgo(self):
        return StandAloneAlgo(self._log, self._mode)
    
    def getPhyAlgo(self):
        return StandAloneAlgo(self._phy, self._mode)

def fromAlgoName(algoName):
    for AlgoClass in [CascadeAlgo, StandAloneAlgo]:
        al = AlgoClass.fromAlgoName(algoName)
        if al is not None:
            return al
    raise RuntimeError("invalid algorithm name: '{}'".format(algoName))

def fromFormatName(formatName):
    for AlgoClass in [CascadeAlgo, StandAloneAlgo]:
        al = AlgoClass.fromFormatName(formatName)
        if al is not None:
            return al
    raise RuntimeError("invalid format name: '{}'".format(formatName))
