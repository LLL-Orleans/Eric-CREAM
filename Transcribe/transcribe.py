################
# All these functions have been designed by Cécile Macaire
# https://github.com/macairececile/ASR-QbE-creole/blob/main/src/lm/create_kenLM.py
# Some functions have been modified by Dr. Éric Le Ferrand
################

# -*- coding: utf-8 -*-
# ----------- Libraries ----------- #
from argparse import ArgumentParser, RawTextHelpFormatter
import difflib
from datasets import load_dataset, load_metric
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2CTCTokenizer
import torch
import numpy as np
import json
from ctcdecode import CTCBeamDecoder
import operator
import re
import torchaudio
import librosa
import preprocessing_creole as prep
from pyannote.audio import Pipeline
import soundfile as sf
from praatio import tgio
from auditok import split
from transformers import WhisperForConditionalGeneration

cer_metric = load_metric('cer')
wer_metric = load_metric('wer')


# ----------- Load the data, the model, tokenizer, processor, process the data ----------- #
def load_model(model_path):
    # Call the fine-tuned model
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    return processor, tokenizer


def load_data(file):
    na_test = load_dataset('csv', data_files=[file], delimiter='\t')
    na_test = na_test['train']
    return na_test


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(
        "/home/getalp/macairec/Bureau/Creole/guadeloupean/clips/" + batch["path"])
    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    batch["target_text"] = batch["sentence"]
    return batch




# ----------- Beam search decoding ----------- #
# Decoding with https://github.com/parlance/ctcdecode library
def beam_search_decoder_lm(processor, tokenizer, logits, lm, alpha, beta):
    vocab = tokenizer.convert_ids_to_tokens(range(0, processor.tokenizer.vocab_size))
    space_ix = vocab.index('|')
    vocab[space_ix] = ' '
    # loaded_lm = kenLM.Model(lm)
    ctcdecoder = CTCBeamDecoder(vocab,
                                model_path=lm,
                                alpha=alpha,
                                beta=beta,
                                cutoff_top_n=300,
                                cutoff_prob=1.0,
                                beam_width=100,
                                num_processes=4,
                                blank_id=processor.tokenizer.pad_token_id,
                                log_probs_input=True
                                )

    beam_results, beam_scores, timesteps, out_lens = ctcdecoder.decode(logits)

    # beam_results - Shape: BATCHSIZE x N_BEAMS X N_TIMESTEPS A batch containing the series
    # of characters (these are ints, you still need to decode them back to your text) representing
    # results from a given beam search. Note that the beams are almost always shorter than the
    # total number of timesteps, and the additional data is non-sensical, so to see the top beam
    # (as int labels) from the first item in the batch, you need to run beam_results[0][0][:out_len[0][0]].
    beam_string = "".join(vocab[n] for n in beam_results[0][0][:out_lens[0][0]])
    # per = loaded_lm.perplexity(beam_string)
    # timesteps : BATCHSIZE x N_BEAMS : the timestep at which the nth output character has peak probability.
    # Can be used as alignment between the audio and the transcript.
    alignment = list()
    for i in range(0, out_lens[0][0]):
        alignment.append([beam_string[i], int(timesteps[0][0][i])])
    return beam_string
# Decoding with https://github.com/parlance/ctcdecode library
def beam_search_decoder_lm(processor, tokenizer, logits, lm, alpha, beta):
    vocab = tokenizer.convert_ids_to_tokens(range(0, processor.tokenizer.vocab_size))
    space_ix = vocab.index('|')
    vocab[space_ix] = ' '
    # loaded_lm = kenLM.Model(lm)
    ctcdecoder = CTCBeamDecoder(vocab,
                                model_path=lm,
                                alpha=alpha,
                                beta=beta,
                                cutoff_top_n=300,
                                cutoff_prob=1.0,
                                beam_width=100,
                                num_processes=4,
                                blank_id=processor.tokenizer.pad_token_id,
                                log_probs_input=True
                                )

    beam_results, beam_scores, timesteps, out_lens = ctcdecoder.decode(logits)

    # beam_results - Shape: BATCHSIZE x N_BEAMS X N_TIMESTEPS A batch containing the series
    # of characters (these are ints, you still need to decode them back to your text) representing
    # results from a given beam search. Note that the beams are almost always shorter than the
    # total number of timesteps, and the additional data is non-sensical, so to see the top beam
    # (as int labels) from the first item in the batch, you need to run beam_results[0][0][:out_len[0][0]].
    beam_string = "".join(vocab[n] for n in beam_results[0][0][:out_lens[0][0]])
    # per = loaded_lm.perplexity(beam_string)
    # timesteps : BATCHSIZE x N_BEAMS : the timestep at which the nth output character has peak probability.
    # Can be used as alignment between the audio and the transcript.
    alignment = list()
    for i in range(0, out_lens[0][0]):
        alignment.append([beam_string[i], int(timesteps[0][0][i])])
    return beam_string

def french_ASR(output, signal):
    import whisper
    from pydub import AudioSegment
    # wh_model = whisper.load_model("large")
    wh_model = WhisperForConditionalGeneration.from_pretrained("/home/getalp/leferrae/post_doc/codeSwitch/code_switch/checkpoint-100/")
    entryList_french = []
    all_seg = AudioSegment.from_wav(args.wav)
    for speech in output.get_timeline().support():
        seg = all_seg[speech.start*1000: speech.end*1000]
        seg.export("temp.wav", format="wav")
        results = wh_model.transcribe("temp.wav", language="fr")
        print(results["text"])

        interval = tgio.Interval(speech.start, speech.end, results["text"])
        entryList_french.append(interval)
    return entryList_french
def transcribe_long_lm(args):
    # args.french = False
    args.noise = False
    tg = tgio.Textgrid()
    model_gwad = "/home/getalp/leferrae/post_doc/model_w2v/out/out_60/checkpoint-3600/"
    processor, tokenizer = load_model(model_gwad)
    vocab = tokenizer.convert_ids_to_tokens(range(0, processor.tokenizer.vocab_size))
    space_ix = vocab.index('|')
    vocab[space_ix] = ' '
    entryList = []
    model = Wav2Vec2ForCTC.from_pretrained(model_gwad).to("cpu")
    signal, sr = sf.read(args.wav)

    dur = (len(signal)/sr)

    ctcdecoder = CTCBeamDecoder(vocab,
                                model_path=args.lm,
                                alpha=args.alpha,
                                beta=args.beta,
                                cutoff_top_n=300,
                                cutoff_prob=1.0,
                                beam_width=100,
                                num_processes=4,
                                blank_id=processor.tokenizer.pad_token_id,
                                log_probs_input=True
                                )

    if args.VAD=="pyannote":
        pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                        use_auth_token="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")
        output = pipeline(args.wav)
        if args.french == True:
            entryList_french = french_ASR(output, signal)
            french_tier = tgio.IntervalTier("French", entryList_french, minT=0, maxT=dur)
            tg.addTier(tier=french_tier)
        for speech in output.get_timeline().support():
            seg = signal[int(speech.start*sr):int(speech.end*sr)]
            input_dict = processor(seg, return_tensors="pt", padding=True, sampling_rate=16000)
            logits = model(input_dict.input_values.to("cpu")).logits

            beam_results, beam_scores, timesteps, out_lens = ctcdecoder.decode(logits)
            preds = "".join(vocab[n] for n in beam_results[0][0][:out_lens[0][0]])

            # preds = beam_search_decoder_lm(processor, tokenizer, logits, args.lm, args.alpha, args.beta)
            interval = tgio.Interval(speech.start, speech.end, preds)
            entryList.append(interval)
        
    
    elif args.VAD=="auditok":
        region = split(args.wav, eth=70, aw=0.01)
        for i, r in enumerate(region):
            seg = signal[int(r.meta.start*sr):int(r.meta.end*sr)]
            input_dict = processor(seg, return_tensors="pt", padding=True, sampling_rate=16000)
            logits = model(input_dict.input_values.to("cpu")).logits

            preds = beam_search_decoder_lm(processor, tokenizer, logits, args.lm, args.alpha, args.beta)
            interval = tgio.Interval(r.meta.start, r.meta.end, preds)
            entryList.append(interval)

    if args.noise==True:
        pipeline_noise = Pipeline.from_pretrained("pyannote/overlapped-speech-detection",
                                        use_auth_token="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")
        entryList_noise = []
        output = pipeline_noise(args.wav)



        for speech in output.get_timeline().support():
            interval = tgio.Interval(speech.start, speech.end, "<Noise>")
            entryList_noise.append(interval)
        
        noise_tier = tgio.IntervalTier("Noise", entryList_noise, minT=0, maxT=dur)
        tg.addTier(tier=noise_tier)
    

        
    
    text = tgio.IntervalTier("Text", entryList, minT=0, maxT=dur)
    tg.addTier(tier=text)
    
    tg.save(args.wav.replace(".wav", ".TextGrid"))

def decoding_lm(args):
    processor, tokenizer, na_test, na_test_ref = pipeline(args.model, args.test)
    preds_lm = []
    refs = []
    for i in range(len(na_test)):
        # load the saved logits to generate the prediction of the model
        logits = torch.load(args.model + '/logits/logits_' + str(i) + '.pt', map_location=torch.device('cpu'))
        preds_lm.append(
            beam_search_decoder_lm(processor, tokenizer, logits, args.lm, args.alpha, args.beta))
        refs.append(na_test['sentence'][i])
        # ------ save all LM predictions in a csv file ------ #
    df_lm = pd.DataFrame({'Reference': refs, 'Prediction': preds_lm})
    df_lm.to_csv(args.model + 'results_decode_lm.csv', index=False, sep='\t')


# ----------- Arguments ----------- #
parser = ArgumentParser(description="Generate predictions with a kenLM language model from a Wav2Vec2.0 model and save them.", formatter_class=RawTextHelpFormatter)

parser.add_argument('--wav', type=str, required=True,
                               help="wav file to transcribe")
parser.add_argument('--VAD', type=str, required=True,
                               help="which VAD between auditok and pyannote")
parser.add_argument('--lm', type=str, required=True,
                               help="Word Ken language model.")
parser.add_argument('--noise', type=bool, required=True,
                               help="do we apply noise detection")
parser.add_argument('--french', type=bool, required=True,
                               help="do we apply French ASR")
parser.add_argument('--alpha', type=int, required=True,
                               help="alpha lm parameter.")
parser.add_argument('--beta', type=int, required=True,
                               help="beta lm parameter.")

parser.set_defaults(func=transcribe_long_lm)
args = parser.parse_args()
args.func(args)