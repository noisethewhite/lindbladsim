from typing import TypeAlias
from numpy import float64, floating, dtype, ndarray, complexfloating
from numpy.typing import NDArray


F64Array:      TypeAlias = NDArray[float64]
FloatArray:    TypeAlias = NDArray[floating]
ComplexArray:  TypeAlias = NDArray[complexfloating]
FloatMatrix:   TypeAlias = ndarray[tuple[int, int], dtype[floating]]
TimeVector:    TypeAlias = ndarray[tuple[int], dtype[floating]]
ComplexVector: TypeAlias = ndarray[tuple[int], dtype[complexfloating]]
