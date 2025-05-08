import VuVoPy as vp
import pandas as pd

# Users should replace the file_path with their own file path
file_path = "user_path_here.wav"

durmad = vp.durmad(file_path)
durmed = vp.durmed(file_path)
duv = vp.duv(file_path)
hnr = vp.hnr(file_path)
jitter = vp.jitterPPQ(file_path)
mpt = vp.mpt(file_path)
ppr =vp.ppr(file_path)
relf0sd = vp.relF0SD(file_path)
relf1sd = vp.relF1SD(file_path)
relf2sd = vp.relF2SD(file_path)
relseosd = vp.relSEOSD(file_path)
shimmer = vp.shimmerAPQ(file_path)
spir = vp.spir(file_path)

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