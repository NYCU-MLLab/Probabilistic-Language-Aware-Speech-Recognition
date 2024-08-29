import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.specaug.specaug import SpecAug


class OpenAIWhisperEncoder(AbsEncoder):
    """Transformer-based Speech Encoder from OpenAI's Whisper Model:

    URL: https://github.com/openai/whisper
    """
    # -------------- edit here ------------------
    def __init__(
        self,
        input_size: int = 1,
        dropout_rate: float = 0.0,
        whisper_model: str = "small",
        download_dir: str = None,
        use_specaug: bool = False,
        specaug_conf: Union[dict, None] = None,
        do_pad_trim: bool = False,
        adapter: str = "",
        r: int = 4,
        sampling:bool = False, 
        sampling_loc: Optional[str] = None,
        lang_rescore: Optional[str] = 'none',
    ):
        try:
            # import whisper
            from whisper import whisper
            from whisper.audio import HOP_LENGTH, N_FFT, N_SAMPLES
            N_MELS = 80
        except Exception as e:
            print("Error: whisper is not properly installed.")
            print(
                "Please install whisper with: cd ${MAIN_ROOT}/tools &&",
                "./installers/install_whisper.sh",
            )
            raise e

        assert check_argument_types()
        super().__init__()

        self.n_fft = N_FFT
        self.win_length = N_FFT
        self.hop_length = HOP_LENGTH
        self.n_mels = N_MELS
        self.lang_rescore = lang_rescore
        self.sampling = sampling

        self.mel_filters = whisper.audio.mel_filters

        # note that originally Whisper doesn't use dropouts
        self.dropout = torch.nn.Dropout(dropout_rate)

        assert whisper_model in whisper.available_models()
        # ----------------- edit here -------------------
        _model = whisper.load_model(whisper_model, download_root=download_dir, 
                                    adapter=adapter, r=r, sampling=sampling,
                                    sampling_loc=sampling_loc, lang_rescore=self.lang_rescore)
        self.encoders = copy.deepcopy(_model.encoder)
        self.encoders.train()

        del _model

        if use_specaug:
            self.specaug = SpecAug(**specaug_conf)
        else:
            self.specaug = None

        self.do_pad_trim = do_pad_trim
        self.pad_samples = N_SAMPLES

    def output_size(self) -> int:
        return self.encoders.ln_post.normalized_shape[-1]

    def pad_or_trim(
        self,
        array: torch.Tensor,
        length: int,
        axis: int = -1,
    ) -> torch.Tensor:
        """Pad or trim the audio array to N_SAMPLES.

        Used in zero-shot inference cases.
        """
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length).to(array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])

        return array

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> torch.Tensor:
        """Use log-mel spectrogram computation native to Whisper training"""
        window = torch.hann_window(self.win_length).to(audio.device)
        stft = torch.stft(
            audio, self.n_fft, self.hop_length, window=window, return_complex=True
        )

        # whisper deletes the last frame by default (Shih-Lun)
        magnitudes = stft[..., :-1].abs() ** 2

        filters = self.mel_filters(audio.device, self.n_mels)
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        if ilens is not None:
            olens = ilens // self.hop_length
        else:
            olens = None

        log_spec = torch.maximum(
            log_spec,
            log_spec.view(audio.size(0), -1).max(dim=-1)[0][:, None, None] - 8.0,
        )
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec, olens

    def whisper_encode(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor = None,
        kl_beta: float = 0.0,
    ) -> torch.Tensor:
        x = F.gelu(self.encoders.conv1(input))
        x = F.gelu(self.encoders.conv2(x))
        x = x.permute(0, 2, 1)

        n_frames = x.size(1)
        max_pos = self.encoders.positional_embedding.size(0)
        if n_frames <= max_pos:
            x = (x + self.encoders.positional_embedding[: x.size(1), :]).to(x.dtype)
        else:
            # due to positional encoding, audios >30 sec won't be accepted
            x = x[:, :max_pos, :] + self.encoders.positional_embedding

        x = self.dropout(x)
        po_list = []
        p_list = []
        if kl_beta != 0.0:
            for layer, block in enumerate(self.encoders.blocks):
                x, _, po_tuple, p_tuple = block(x, kl_beta=kl_beta)
                if layer < len(self.encoders.blocks) - 1:
                    x = self.dropout(x)
                # po_list.append(po_tuple)
                # p_list.append(p_tuple)
        else:
            for layer, block in enumerate(self.encoders.blocks):
                x, _, _, _ = block(x)
                if layer < len(self.encoders.blocks) - 1:
                    x = self.dropout(x)

        x = self.encoders.ln_post(x)
        # if self.lang_rescore == 'vib' and kl_beta != 0.0:
            # mu = self.encoders.mean_adapter(x)
            # std = F.relu(self.encoders.var_adapter(x))
            # eps = torch.randn_like(std)
            # x = mu + std * eps
            # x = self.encoders.vib_ln(x)

            # mu_p = self.encoders.mu_p.unsqueeze(0).unsqueeze(0)
            # std_p = self.encoders.std_p.unsqueeze(0).unsqueeze(0)
            # mu_p = mu_p.expand(mu.shape)
            # std_p = std_p.expand(std.shape)
            # po_list.append((mu, std))
            # p_list.append((mu_p, F.relu(std_p)))
            


        if ilens is not None:
            olens = (
                1
                + (
                    ilens
                    - self.encoders.conv2.kernel_size[0]
                    + 2 * self.encoders.conv2.padding[0]
                )
                // self.encoders.conv2.stride[0]
            )
            olens = torch.clamp(olens, max=max_pos)
        else:
            olens = None

        return x, olens, po_list, p_list

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        kl_beta: float = 0.0,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.do_pad_trim:
            xs_pad = self.pad_or_trim(xs_pad, self.pad_samples)

        feats, feats_lens = self.log_mel_spectrogram(xs_pad, ilens)

        if self.specaug is not None and self.encoders.training:
            feats = torch.transpose(feats, 1, 2)
            feats, feats_lens = self.specaug(feats, feats_lens)
            feats = torch.transpose(feats, 1, 2)
     
        xs_pad, olens, po_list, pr_list = self.whisper_encode(feats, feats_lens, kl_beta)
        # if kl_beta != 0.0:
        #     noise = torch.randn_like(feats)
        #     x_noise, _, _, _ = self.whisper_encode(feats+0.1*noise, feats_lens, kl_beta)
        #     a = torch.zeros(x_noise.shape)
        #     return xs_pad, olens, None, [(x_noise, a)], [(xs_pad, a)]
        return xs_pad, olens, None, po_list, pr_list

