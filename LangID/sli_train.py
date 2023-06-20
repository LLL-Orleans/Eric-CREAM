import pickle

from argparse import ArgumentParser
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle

from helpers import get_sli_df, get_sb_encoder, add_sbemb_cols, colsplit_feats_labels

sli_df = get_sli_df("./sub_bin/train/")
print("Extracting features...")
sli_df = add_sbemb_cols(sli_df, sb_encoder=get_sb_encoder())
feats, labels = colsplit_feats_labels(sli_df)
feats, labels = shuffle(feats, labels, random_state=0)

print("Fitting classifier...")
clf = LogisticRegression(random_state=0, max_iter=5000).fit(feats, labels)

pickle.dump(clf, open("./lang_id_bin.pkl", 'wb'))
print(f"Saved classifier to ./lang_id_bin.pkl")