import torchaudio
import _pickle as pickle
from speechbrain.pretrained import EncoderClassifier
from praatio import tgio
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.manifold import TSNE
import os
from tqdm import tqdm
sb_embd = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="./pretrained/")
sli_clf = pickle.load(open("lang_id_en.pkl", 'rb'))

tg = tgio.openTextgrid("lang_id.TextGrid")
tier=tg.tierNameList[2]
transcs = tg.tierDict[tg.tierNameList[1]].entryList
embs = []
labels = []
wav_path = "./sub_All/"
partition = "train/"
path_lang1 = wav_path+partition+"en/"
path_lang2 = wav_path+partition+"gw/"
path_lang3 = wav_path+partition+"fr/"
wav_paths = [path_lang1+x for x in os.listdir(path_lang1)]+[path_lang2+x for x in os.listdir(path_lang2)]+[path_lang3+x for x in os.listdir(path_lang3)]
for wav_file in tqdm(wav_paths[0:100]):
    waveform, sample_rate = torchaudio.load(wav_file)
    x = sb_embd.encode_batch(waveform).reshape((1, 256))
    if "gw" in wav_file:
        lang = "gw"
    elif "en" in wav_file:
        lang = "en"
    elif "fr" in wav_file:
        lang = "fr"
    embs.append(np.array(x[0]))
    labels.append(lang)

print(np.array(embs).shape, np.array(labels).shape)
X_embedded = TSNE(n_components=2).fit_transform(np.array(embs))

df_embeddings = pd.DataFrame(X_embedded)
df_embeddings = df_embeddings.rename(columns={0:'x',1:'y'})
df_embeddings = df_embeddings.assign(label=labels)

fig = px.scatter(
    df_embeddings, x='x', y='y',
    color='label', labels={'color': 'label'},
    hover_data=['label'], title = 'Language Embedding Visualization')
fig.show()