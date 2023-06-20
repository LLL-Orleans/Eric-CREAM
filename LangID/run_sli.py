import torchaudio
import _pickle as pickle
from speechbrain.pretrained import EncoderClassifier
from praatio import tgio
import soundfile as sf
from pydub import AudioSegment
import os
import random

lang_lex = "fr"

sb_embd = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="./pretrained/")
sli_clf = pickle.load(open("lang_id_fr.pkl", 'rb'))
en_root = "/home/getalp/leferrae/post_doc/prep_gwad/sub_{}/dev/{}/".format(lang_lex.upper(), lang_lex)
gw_root = "/home/getalp/leferrae/post_doc/prep_gwad/sub_{}/dev/gw/".format(lang_lex.upper())
data = []
for wav in os.listdir(en_root):
    data.append({"lang" : "fr", "path" : en_root+wav})
for wav in os.listdir(gw_root):
    data.append({"lang" : "gw", "path" : gw_root+wav})

precision_gw = 0
precision_hr = 0
tot_hr = 0
tot_gw = 0
random.shuffle(data)
for elt in data:
    waveform, sample_rate = torchaudio.load(elt["path"])
    emb  = sb_embd.encode_batch(waveform).reshape((1, 256))
    lang = sli_clf.predict(emb)[0]
    if lang==lang_lex:
        tot_hr+=1
        if lang == elt["lang"]:
            precision_hr+=1
    else:
        tot_gw+=1
        if lang == elt["lang"]:
            precision_gw+=1
    print("prediction : {}, gold: {}".format(lang, elt["lang"]))

print("precision {}: {}%; precision gw: {}%".format(lang_lex, precision_hr/tot_hr*100, precision_gw/tot_gw*100))