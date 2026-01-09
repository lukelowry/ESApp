

class ESAPlusError(Exception):
    '''Base exception class for ESA++ library errors'''
    pass


class GridObjDNE(ESAPlusError):
    '''Describes a data query failure'''
    pass

class FieldDataException(ESAPlusError):
    pass

class AuxParseException(ESAPlusError):
    pass

class ContainerDeletedException(ESAPlusError):
    pass

'''Observable Exceptions'''

class PowerFlowException(ESAPlusError):
    '''Raised When Power Flow Error Occurs'''
    pass

class BifurcationException(PowerFlowException):
    '''Raised when bifurcation is suscpected'''
    pass 

class DivergenceException(PowerFlowException): # TODO in use?
    pass 

class GeneratorLimitException(PowerFlowException):
    '''Raised when a generator has exceed a limit'''
    pass 

''' GIC Exceptions '''

class GICException(ESAPlusError):
    pass 

