.. VuVoPy documentation master file, created by sphinx-quickstart

VuVoPy Documentation
====================

**VuVoPy** is a Python library for extracting acoustic features from speech signals, 
specifically developed for biomedical voice analysis and Parkinson’s disease research.

This documentation includes:

- 📦 Module overviews
- 📊 Feature extraction functions
- 🧠 Usage examples
- 🛠 Developer reference


Getting Started
---------------

To install VuVoPy in development mode:

.. code-block:: bash

    pip install VuVoPy

To use it in Python:

.. code-block:: python

    import VuVoPy as vp
    value = vp.durmad(my_signal)


.. toctree::
   :maxdepth: 4
   :caption: Contents:

   modules


API Reference
-------------

Below you can find the automatically generated API reference for each module and submodule
in the package. Click through to view detailed docstrings, parameters, and return types.


Here is a simple example of using VuVoPy to compute speech parameters:

.. code-block:: python

   import VuVoPy as vp
   import pandas as pd

   # Users should replace the file_path with their own file path
   file_path = "signal.wav"
   durmad = vp.durmad(file_path,winlen=512,winover=256,wintype='hamm')
   durmed = vp.durmed(file_path,winlen=512,winover=256,wintype='hamm')
   duv = vp.duv(file_path,winlen=512,winover=256,wintype='hamm')
   hnr = vp.hnr(file_path)
   jitter = vp.jitterPPQ(file_path)
   mpt = vp.mpt(file_path,winlen=512,winover=256,wintype='hamm')
   ppr =vp.ppr(file_path,winlen=512,winover=256,wintype='hamm')
   relf0sd = vp.relF0SD(file_path)
   relf1sd = vp.relF1SD(file_path,winlen=512,winover=256,wintype='hamm')
   relf2sd = vp.relF2SD(file_path)
   relseosd = vp.relSEOSD(file_path,winlen=512,winover=256,wintype='hamm')
   shimmer = vp.shimmerAPQ(file_path)
   spir = vp.spir(file_path, winlen=512,winover=256,wintype='hamm')

   data = {
       "durmad": [durmad],
       "durmed": [durmed],
       "duv": [duv],
       "hnr": [hnr],
       "jitter": [jitter],
       "mpt": [mpt],
       "ppr": [ppr],
       "relf0sd": [relf0sd],
       "relf1sd": [relf1sd],
       "relf2sd": [relf2sd],
       "relseosd": [relseosd],
       "shimmer": [shimmer],
       "spir": [spir]
   }

   df = pd.DataFrame(data)
   print(df)
