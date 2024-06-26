# Multilingual Lyrics Transcription

This repository consists of code of the following paper:

Jiawen Huang, Emmanouil Benetos, “**Towards Building an End-to-End Multilingual Automatic Lyrics Transcription Model**”, 
32th European Signal Processing Conference, Lyon, France, 2024. 
[Link](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/97337/Huang%20Towards%20Building%20an%202024%20Accepted.pdf?sequence=2&isAllowed=y)

## Dependencies

This repo is written in python 3.x using the speechbrain toolkit. To install the dependencies with conda, run the following command:

```
# create and activate a new conda environment
conda create -n mtlalt python=3.9
conda activate mtlalt

# install pytorch with the right cuda version
pip install torch==1.12.1+cu116 torchaudio -f https://download.pytorch.org/whl/cu116/torch_stable.html

# install the rest of the dependencies
pip install -r requirements.txt
```

## Preparing Data

### Datasets

The **DALI v2** and **MulJam v1** are used for training in this work. The **MultiLang Jamendo v1** is used for evaluation

See instructions on how to get the datasets: 

DALI v2: [https://github.com/gabolsgabs/DALI](https://github.com/gabolsgabs/DALI). 

MulJam v2: [https://github.com/a43992899/MARBLE-Benchmark/issues/13](https://github.com/a43992899/MARBLE-Benchmark/issues/13)

MultiLang Jamendo v1.1: [https://github.com/f90/jamendolyrics](https://github.com/f90/jamendolyrics)

All songs are **source-separated** and **segmented into utterances**. The utterances are organized as the following structure:

```
$muljam_path
├── train
├── valid
$dali_path
├── train
├── valid
jamendolyrics
├── ...
```

Notes on version changes of the datasets: 
1. In the paper we were using the MulJam v1 version. MulJam v2 is slightly different in the train/valid split. 
MulJam v2 is recommended as it removes overlap songs in MultiLang Jamendo.
2. One song from the MultiLang Jamendo dataset has been removed from the official repository.
3. We will report the results on the MultiLang Jamendo v1.1, training with DALI v2 and MulJam v2 for future reference.

### Data splits

Data splits are placed under **./data/**, in the speechbrain style. 


[training splits are not released yet.]

### Tokenizers

Tokenizer References (huggingface: vocab.json):
```
English (en):       facebook/wav2vec2-large-960h-lv60-self
French (fr):        facebook/wav2vec2-large-xlsr-53-french
Spanish (es):       facebook/wav2vec2-large-xlsr-53-spanish
German (de):        facebook/wav2vec2-large-xlsr-53-german
Italian (it):       facebook/wav2vec2-large-xlsr-53-italian
Russian (ru):       jonatasgrosman/wav2vec2-large-xlsr-53-russian
```

## Training and Inference

```
# define the paths to the datasets
dali_path=/path/to/dali
muljam_path=/path/to/muljam

# multilingual
# python train_mix.py hparams/mel_mix.yaml --muljam_data_folder $muljam_path --dali_data_folder $dali_path

# monolingual (French for example)
python train_mix.py hparams/lang/mel_fr.yaml --muljam_data_folder $muljam_path --dali_data_folder $dali_path

# language-informed (conditioning on both encoder and decoder)
# python train_lang_inform.py hparams/lang-cond/both_lang_inform.yaml --muljam_data_folder $muljam_path --dali_data_folder $dali_path

# self-conditioning
# python train_lang_self.py hparams/lang-cond/self_cond.yaml --muljam_data_folder $muljam_path --dali_data_folder $dali_path
```

## References

[1] speechbrain transformer ASR recipe: [https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriSpeech/ASR/transformer](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriSpeech/ASR/transformer)

[2] Xiaoxue Gao, Chitralekha Gupta and Haizhou Li. "PoLyScriber: Integrated Fine-Tuning of Extractor and Lyrics Transcriber for Polyphonic Music." IEEE/ACM Transactions on Audio, Speech, and Language Processing, Vol 31, pp. 1968-1981, 2023.

[3] Yuan, Ruibin, Yinghao Ma, Yizhi Li, Ge Zhang, Xingran Chen, Hanzhi Yin, Yiqi Liu et al. "MARBLE: Music audio representation benchmark for universal evaluation." Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

[4] Ou, Longshen, Xiangming Gu, and Ye Wang. "Transfer learning of wav2vec 2.0 for automatic lyric transcription." Proceedings of the 23rd International Society for Music Information Retrieval Conference, {ISMIR} 2022, Bengaluru, India, December 4-8, 2022.

[//]: # (## Cite this work)
