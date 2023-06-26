This is a set of scripts to create acoustic words embeddings for spoken terms detection purposes

cae_emb is a notebook including batching and training functions to create words embeddings based on a corresponding AutoEncoder framework with keras layers. The training was successful but did not provide better performances for spoken term detection using DTW

classifer.py is a script to create a spoken terms classifer using a Triplet loss. This script never actually worked, the training failed with probably the gradient exploding. Dig it up if you feel like it

model.py is the same kind of idea that classifer...

train_ae is a script to train an AutoEndoer for Spoken Term Detection purposes. same comment that cae_emb

