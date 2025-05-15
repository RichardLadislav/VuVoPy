import VuVoPy as vp
import pandas as pd

# Users should replace the file_path with their own file path
#file_path = "user_path_here.wav"
file_path = "C://Users//Richard Ladislav//Desktop//final countdown//DP-knihovna pro parametrizaci reci - kod//short_test_database//recordings//K1021//K1021_7.1-3-a_1.wav"

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
# 2) print LaTeX tabular code
#latex_table = df.to_latex(
#    index=False,
#    float_format="%.4f",          # adjust number formatting
#    caption="Extracted Voice Parameters",
#    label="tab:voice_params",
#    column_format="l" + "r" * (df.shape[1] - 1)  # left-align first, right-align rest
#)
#print(latex_table)