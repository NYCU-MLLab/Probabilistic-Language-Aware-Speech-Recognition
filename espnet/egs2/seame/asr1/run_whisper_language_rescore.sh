#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=13

train_set="train"
valid_set="valid"
# test_sets="devman devsge test_ascend test_ntut test_aishell"
test_sets="devman devsge"

lm_config=conf/tuning/train_lm_transformer.yaml
asr_config=conf/whisper/train_asr_whisper_language_rescore.yaml
inference_config=conf/whisper/decode_asr_whisper.yaml

freeze_param="adapter"
use_lm=false
use_wordlm=false
# pre_trained_weight=/home/espnet/egs2/seame/asr1/exp/asr_seame_whisper_small_zh_en_adapter_encoder_nocsloss/valid.acc.ave.pth
./asr.sh \
    --ngpu 1 \
    --nj 5 \
    --gpu_inference true \
    --inference_nj 1 \
    --ngpu 1 \
    --audio_format "flac.ark" \
    --feats_type raw \
    --stage ${stage} \
    --stop_stage ${stop_stage} \
    --token_type whisper_multilingual \
    --feats_normalize '' \
    --cleaner whisper_basic \
    --use_lm ${use_lm}                  \
    --use_word_lm ${use_wordlm}         \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --min_wav_duration 1 \
    --max_wav_duration 30 \
    --asr_speech_fold_length 800 \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text.eng.bpe" \
    --asr_tag "asr_seame_language_rescore" \
    --score_opts "-e utf-8 -c NOASCII" \
    --freeze_param "${freeze_param}" \
    "$@"
