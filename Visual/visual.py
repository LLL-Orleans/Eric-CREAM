import torchaudio
import _pickle as pickle
from speechbrain.pretrained import EncoderClassifier
from praatio import tgio
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn.manifold import TSNE


sb_embd = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="./pretrained/")
sli_clf = pickle.load(open("lang_id_en.pkl", 'rb'))

tg = tgio.openTextgrid("lang_id.TextGrid")
tier=tg.tierNameList[2]
transcs = tg.tierDict[tg.tierNameList[1]].entryList
waveform, sr = torchaudio.load("/home/getalp/leferrae/post_doc/corpora/Collect_Nov22/1July22/1July2022.wav")
embs = []
labels = []
for i, tg_part in enumerate(tg.tierDict[tier].entryList):
    startWord = float(tg_part[0])
    endWord = float(tg_part[1])
    if (endWord-startWord)>2:
        seg = waveform[:,int(startWord*sr): int(endWord*sr)]
        emb  = sb_embd.encode_batch(seg).reshape((1, 256))
        print(np.array(emb[0]).shape)
        lang = sli_clf.predict(emb)[0]
        embs.append(np.array(emb[0]))
        labels.append(tg_part[2])

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