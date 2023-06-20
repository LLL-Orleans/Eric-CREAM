import torchaudio
import _pickle as pickle
from speechbrain.pretrained import EncoderClassifier
from praatio import tgio
import soundfile as sf
from pydub import AudioSegment
import os
import random
import soundfile as sf


precision_gw = 0
precision_fr = 0
precision_en = 0
recall_gw = 0
recall_fr = 0
recall_en = 0
totpres_gw, totpres_en, totpres_fr = 0,0,0
min_length = 2

sb_embd = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="./pretrained/")
sli_clf = pickle.load(open("lang_id_all.pkl", 'rb'))

tg = tgio.openTextgrid("lang_id.TextGrid")
tier=tg.tierNameList[2]
transcs = tg.tierDict[tg.tierNameList[1]].entryList
waveform, sr = torchaudio.load("/home/getalp/leferrae/post_doc/corpora/Collect_Nov22/1July22/1July2022.wav")
tot_fr = len([x for x in tg.tierDict[tier].entryList if x[2]=="fr" and float(x[1])-float(x[0])>min_length])
tot_en = len([x for x in tg.tierDict[tier].entryList if x[2]=="en" and float(x[1])-float(x[0])>min_length])
tot_gw = len([x for x in tg.tierDict[tier].entryList if x[2]=="gw" and float(x[1])-float(x[0])>min_length])

for i, tg_part in enumerate(tg.tierDict[tier].entryList):
    startWord = float(tg_part[0])
    endWord = float(tg_part[1])
    if (endWord-startWord)>min_length:
        seg = waveform[:,int(startWord*sr): int(endWord*sr)]
        emb  = sb_embd.encode_batch(seg).reshape((1, 256))
        lang = sli_clf.predict(emb)[0]
        if lang=="fr":
            totpres_fr+=1
            if lang == tg_part[2]:
                precision_fr+=1
        elif lang=="gw" :
            totpres_gw+=1
            if lang  == tg_part[2]:
                precision_gw+=1
        elif lang=="en":
            totpres_en+=1
            if lang == tg_part[2]:
                precision_en+=1
        # if lang!=tg_part[2]:
        #     print("prediction : {}, gold: {}".format(lang, tg_part[2]), transcs[i])

print("precision en: {}%; precision fr: {}%; precision gw: {}%".format(precision_en/totpres_en*100, precision_fr/totpres_fr*100, precision_gw/totpres_gw*100))
print("recall en: {}%; recall fr: {}%; recall gw: {}%".format(precision_en/tot_en*100, precision_fr/tot_fr*100, precision_gw/tot_gw*100))

print("precision en: {}/{}; precision fr: {}/{}; precision gw: {}/{}".format(precision_en,totpres_en, precision_fr,totpres_fr, precision_gw,totpres_gw))
print("recall en: {}/{}; recall fr: {}/{}; recall gw: {}/{}".format(precision_en,tot_en, precision_fr,tot_fr, precision_gw,tot_gw))

