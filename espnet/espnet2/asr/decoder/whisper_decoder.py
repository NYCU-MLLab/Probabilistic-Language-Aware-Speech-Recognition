import copy
from typing import Any, List, Tuple, Optional

import torch
from typeguard import check_argument_types

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.scorer_interface import BatchScorerInterface

import re
import string

class OpenAIWhisperDecoder(AbsDecoder, BatchScorerInterface):
    """Transformer-based Speech-to-Text Decoder from OpenAI's Whisper Model:

    URL: https://github.com/openai/whisper
    """
    # ------------ edit here -----------------
    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        whisper_model: str = "small",
        download_dir: str = None,
        adapter: str = "",
        r: int = 4,
        sampling:bool = False, 
        sampling_loc: Optional[str] = None ,
        lang_rescore: str = 'none',
        scale: float = 0.5,
    ):
        try:
            # import whisper
            from whisper import whisper
        except Exception as e:
            print("Error: whisper is not properly installed.")
            print(
                "Please install whisper with: cd ${MAIN_ROOT}/tools && "
                "./installers/install_whisper.sh"
            )
            raise e

        assert check_argument_types()
        super().__init__()

        assert whisper_model in whisper.available_models()
        # ----------------- edit here -------------------
        _model = whisper.load_model(whisper_model, download_root=download_dir, 
                                    adapter=adapter, r=r, sampling=sampling,
                                    sampling_loc=sampling_loc, lang_rescore=lang_rescore)
        self.decoders = copy.deepcopy(_model.decoder)
        attention_dim = self.decoders.token_embedding.embedding_dim

        # note that originally Whisper doesn't use dropouts
        self.dropout = torch.nn.Dropout(dropout_rate)

        # vocab size mismatch -> reinitialize embedding
        # orig vocab size (multilingual): 51865
        # orig vocab size (english): 51864
        if vocab_size != self.decoders.token_embedding.num_embeddings:
            orig_emb_std, orig_emb_mean = torch.std_mean(
                self.decoders.token_embedding.weight
            )
            self.decoders.token_embedding = torch.nn.Embedding(
                vocab_size, attention_dim
            )
            torch.nn.init.normal_(
                self.decoders.token_embedding.weight,
                orig_emb_mean.item(),
                orig_emb_std.item(),
            )

        self.decoders.train()
        del _model

        self.lang_rescore = lang_rescore
        self.scale = scale
        self.token_map = self._create_token_maps()
        # self.lang_indices = torch.vstack([(self.token_map == lang).to(torch.bfloat16) for lang in range(3)])

    def _determine_token_type(self, s):
        # Define the pattern for special tokens
        special_token_pattern = r"<|[A-Z]+|>"

        # Check if the string matches the special token pattern
        # if re.fullmatch(special_token_pattern, s):
        #     return 2    # special token
        if "<|" in s and "|>" in s:
            return 2    # special token
        
        # Check if the string contains only English characters and spaces
        if all(char in string.ascii_letters for char in s):
            return 1    # en

        return 0    # zh
    
    def _create_token_maps(self):
        from transformers import WhisperTokenizer
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="chinese", task="transcribe")
        vocab = tokenizer.get_vocab()
        token_list = list(vocab.keys())
        N = len(token_list)
        token_classes = torch.empty(N, dtype=torch.float16)  # assuming _determine_token_type returns float

        for i, token in enumerate(token_list):
            token_classes[i] = self._determine_token_type(tokenizer.convert_tokens_to_string(token))  # only replace if necessary

        return token_classes


    def add_token_lang_prob_to_token_prob(self, token_prob, lang_prob, scale=1):
        # token_prob [batch_size, length, V]
        # lang_prob [batch_size, length, 3]
        # output [batch_size, length, V]
        data_type = lang_prob.dtype
        lang_indices = torch.vstack([(self.token_map == lang).to(data_type) for lang in range(3)])
        projected_lang_prob = lang_prob @ lang_indices.to(lang_prob.device)
        # projected_lang_prob = token_prob * projected_lang_prob
        projected_lang_prob = token_prob + projected_lang_prob
        combined_prob = token_prob + (scale * projected_lang_prob)
        
        return combined_prob

    def token_language_probability(self, lang_prob):
        # lang_prob [3]
        # output [V]
        # idx 0 = zh, 1 = en, 2 = other
        indices = torch.tensor([lang for lang in self.token_map])
        return lang_prob[indices]

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad:
                input token ids, int64 (batch, maxlen_out)
                if input_layer == "embed"
                input tensor (batch, maxlen_out, #mels) in the other cases
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        # if self.token_map == None:
        #     self.token_map = token_map

        tgt, memory = ys_in_pad, hs_pad
        tgt = (
            self.decoders.token_embedding(tgt)
            + self.decoders.positional_embedding[: tgt.size(1)]
        )
        tgt = self.dropout(tgt)

        x = tgt.to(memory.dtype)
        attention_scores = []
        for layer, block in enumerate(self.decoders.blocks):
            x, attention_map, _, _ = block(x, memory, mask=self.decoders.mask)
            if layer < len(self.decoders.blocks) - 1:
                x = self.dropout(x)
            attention_scores.append(attention_map)
        x = self.decoders.ln(x)
        y = (
            x @ torch.transpose(self.decoders.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        # iseng
        y = torch.log_softmax(y, dim=-1)
        # ------------------------ edit here ----------------------------------
        lang = None
        if self.lang_rescore == 'adapter':
            lang = self.decoders.language_adapter(x)
            # iseng
            lang = torch.softmax(lang, dim=-1)
            y = self.add_token_lang_prob_to_token_prob(y, lang)
            # y = self.decoders.lang_ln(y)
        # ---------------------------------------------------------------------

        # return y, attention_scores, None
        return y, attention_scores, lang
        # return x, ys_in_lens

    def forward_one_step(
        self,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
        memory: torch.Tensor,
        cache: List[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward one step.

        Args:
            tgt: input token ids, int64 (batch, maxlen_out)
            tgt_mask: input token mask,  (batch, maxlen_out)
                      dtype=torch.uint8 in PyTorch 1.2-
                      dtype=torch.bool in PyTorch 1.2+ (include 1.2)
            memory: encoded memory, float32  (batch, maxlen_in, feat)
            cache: cached output list of (batch, max_time_out-1, size)
        Returns:
            y, cache: NN output value and cache per `self.decoders`.
            y.shape` is (batch, maxlen_out, token)
        NOTE (Shih-Lun):
            cache implementation is ignored for now
            for simplicity & correctness
        """

        if(tgt.size(dim=1)>448):
            tgt=tgt[:, :448]

        x = (
            self.decoders.token_embedding(tgt)
            + self.decoders.positional_embedding[: tgt.size(1)]
        )

        x = self.dropout(x)
        x = x.to(memory.dtype)

        attention_scores = []
        for layer, block in enumerate(self.decoders.blocks):
            x, _, _, _ = block(x, memory, mask=self.decoders.mask)
            # attention_scores.append(att_map.cpu())
            if layer < len(self.decoders.blocks) - 1:
                x = self.dropout(x)

        x = self.decoders.ln(x)
        y = x[:, -1]
        y = (
            y @ torch.transpose(self.decoders.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        y = torch.log_softmax(y, dim=-1)
        # ------------------------ edit here ----------------------------------
        if self.lang_rescore == 'adapter':
            lang = self.decoders.language_adapter(x[:, -1])
            # iseng
            lang = torch.softmax(lang, dim=-1)
            y = self.add_token_lang_prob_to_token_prob(y, lang)
            # y = self.decoders.lang_ln(y)
        # ---------------------------------------------------------------------

        # ------------- used for attention map ----------------------
        # if torch.argmax(y).cpu() == 50257: # EOT (?)
        #     print('end')
        # y = torch.log_softmax(y, dim=-1)
        
        return y, None

    def score(self, ys, state, x):
        """Score."""
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), torch.empty(0), x.unsqueeze(0), cache=state  # dummy mask
        )
        return logp.squeeze(0), state

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # batch decoding, dummy mask is passed
        logp, states = self.forward_one_step(ys, torch.empty(0), xs, cache=None)

        return logp, None
