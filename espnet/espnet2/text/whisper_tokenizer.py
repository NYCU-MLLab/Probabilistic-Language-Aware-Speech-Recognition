from typing import Iterable, List

from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer

LANGUAGES_CODE_MAPPING = {
    "noinfo": "english",  # default, English
    "ca": "catalan",
    "cs": "czech",
    "cy": "welsh",
    "de": "german",
    "en": "english",
    "eu": "basque",
    "es": "spanish",
    "fa": "persian",
    "fr": "french",
    "it": "italian",
    "ja": "japanese",
    "jpn": "japanese",
    "ko": "korean",
    "kr": "korean",
    "nl": "dutch",
    "pl": "polish",
    "pt": "portuguese",
    "ru": "russian",
    "tt": "tatar",
    "zh": "chinese",
    "zh-TW": "chinese",
    "zh-CN": "chinese",
    "zh-HK": "chinese",
}


class OpenAIWhisperTokenizer(AbsTokenizer):
    def __init__(self, model_type: str, language: str = "en"):
        assert check_argument_types()

        try:
            # import whisper.tokenizer
            from whisper import whisper
            from transformers import WhisperTokenizer
        except Exception as e:
            print("Error: whisper is not properly installed.")
            print(
                "Please install whisper with: cd ${MAIN_ROOT}/tools && "
                "./installers/install_whisper.sh"
            )
            raise e

        self.model = model_type

        self.language = LANGUAGES_CODE_MAPPING.get(language)
        if self.language is None:
            raise ValueError("language unsupported for Whisper model")

        if model_type == "whisper_en":
            self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False)
        elif model_type == "whisper_multilingual":
            # self.tokenizer = whisper.tokenizer.get_tokenizer(
            #     multilingual=True, language=self.language
            # )
            self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="english", task="transcribe")
        else:
            raise ValueError("tokenizer unsupported:", model_type)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(model_type={self.model}, "
            f"language={self.language})"
        )

    def text2tokens(self, line: str) -> str:
        # return self.tokenizer.tokenize(line, add_special_tokens=False)
        return self.tokenizer._tokenize(line)
        # return self.tokenizer.encode(line)

        # Done in whisper_token_id_converter.py
        # return line

    def tokens2text(self, tokens: Iterable[str]) -> List[str]:
        return self.tokenizer.convert_tokens_to_string(tokens)
        # return self.tokenizer.decode(tokens)

        # Done in whisper_token_id_converter.py
        # return tokens
