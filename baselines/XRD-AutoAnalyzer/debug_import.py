import sys
import os
from autoXRD import quantifier
import inspect

print(f"File: {quantifier.__file__}")
print(f"Signature: {inspect.signature(quantifier.main)}")
