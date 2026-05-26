class SSGError(Exception):
    """
    the base class for all exceptions in SSG4Wann.
    """
    pass

class SpinRotationError(SSGError):
    """
    The exception raised when the spin rotation matrix calculation fails
    """
    pass

class WannierMatchError(SSGError):
    """
    The exception raised when no matching Wannier function is found
    """
    pass

class ConfigParseError(SSGError):
    """
    The exception raised when there is a syntax error in parsing the sg.in configuration file or INCAR
    """
    pass

class AngularMomentumError(SSGError):
    """
    The exception raised when the rotation of Wannier functions fails
    """
    pass

class GroupStructureError(SSGError):
    """
    The exception raised when the symmetry group structure is not as expected
    """
    pass