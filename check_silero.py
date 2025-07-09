#!/usr/bin/env python
import sys
print(sys.path)
try:
    import silero_vad
    print('Silero VAD module found')
    print(dir(silero_vad))
except ImportError as e:
    print(f'Error importing silero_vad: {e}')
