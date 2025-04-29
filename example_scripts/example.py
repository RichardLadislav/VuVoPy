import VuVoPy as vp
import pandas as pd

#file_path = "file_path.wav" # Replace with your actual file path
file_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//short_test_database//recordings//K1021//K1021_10.2-1_1.wav"
#file_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//concept_algorithms_zaloha//vowel_e_test.wav" # Replace with your actual file path
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