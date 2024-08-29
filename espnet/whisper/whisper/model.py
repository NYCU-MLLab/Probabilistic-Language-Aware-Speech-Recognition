import base64
import gzip
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .decoding import decode as decode_function
from .decoding import detect_language as detect_language_function
from .transcribe import transcribe as transcribe_function

import math

@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class Adapter(nn.Module):
    def __init__(self, idim, bottleneck_dim=None) -> None:
        super().__init__()
        # bottleneck_dim = bottleneck_dim if bottleneck_dim else int(idim//4)
        bottleneck_dim = bottleneck_dim if bottleneck_dim else int(idim//5)
        self.model = nn.Sequential(
             #nn.Linear(10, 10),
             nn.Linear(idim, bottleneck_dim),
             nn.GELU(),
             nn.Linear(bottleneck_dim, idim),
        )

    def forward(self,input):
        output = self.model(input)
        return input + output

class LanguageAdapter(nn.Module):
    def __init__(self, idim, bottleneck_dim=None) -> None:
        super().__init__()
        num_classes = 3         # en, zh, and special
        bottleneck_dim = bottleneck_dim if bottleneck_dim else int(idim//4)
        self.model = nn.Sequential(
             nn.Linear(idim, bottleneck_dim),
             nn.GELU(),
             nn.Linear(bottleneck_dim, num_classes),
            #  nn.LayerNorm(num_classes)
        )
       # self.model = nn.Linear(idim, num_classes)

    def forward(self,input):
        output = self.model(input)
        return output

class AdapterSingle(nn.Module):
    def __init__(self, idim, initialization="") -> None:
        super().__init__()
        # self.model = nn.Linear(idim, idim)
        self.model1 = nn.Linear(idim, idim//4)
        self.model2 = nn.Linear(idim//4, idim)
        self.mean_var = "mean" if initialization == "" else "var"
        if initialization == 'zero':
            # nn.init.zeros_(self.model.weight)
            # nn.init.zeros_(self.model.bias)
            nn.init.zeros_(self.model2.weight)
            nn.init.zeros_(self.model2.bias)

    def forward(self,input):
        # output = self.model(input)
        # if self.mean_var == "var":
        #     return F.relu(output)
        # return output
        output = self.model1(input)
        output = self.model2(output)
        if self.mean_var == "var":
            return F.relu(output)
        return output

    
class AdapterReLU(nn.Module):
    def __init__(self, idim, bottleneck_dim=None) -> None:
        super().__init__()
        bottleneck_dim = bottleneck_dim if bottleneck_dim else int(idim//4)
        # bottleneck_dim = bottleneck_dim if bottleneck_dim else int(idim//2)
        # bottleneck_dim = bottleneck_dim if bottleneck_dim else int(idim//5)
        self.lin1 = nn.Parameter(torch.randn(idim, bottleneck_dim))
        self.lin2 = nn.Parameter(torch.zeros(bottleneck_dim, idim))
        # self.bias1 = nn.Parameter(torch.zeros(idim))
        # self.bias2 = nn.Parameter(torch.zeros(bottleneck_dim))
        # self.model = nn.Sequential(
        #      self.lin1,
        #      nn.GELU(),
        #      self.lin2,
        # )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lin1, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.bias1, a=math.sqrt(5))
        nn.init.zeros_(self.lin2)
        # nn.init.zeros_(self.bias2)

    def forward(self,input):
        # result = (input @ self.lin1) + self.bias1
        # result = F.gelu(result)
        # result = (result @ self.lin2) + self.bias2

        result = (input @ self.lin1)
        result = F.gelu(result)
        result = (result @ self.lin2)
        return F.relu(result)
        # return F.softplus(result)
    
class LoRA(nn.Module):
    def __init__(self, idim, alpha=1, r=4) -> None:
        super().__init__()
        self.lora_a = nn.Parameter(torch.randn(r, idim))
        self.lora_b = nn.Parameter(torch.zeros(idim, r))
        self.alpha = alpha
        self.r = r
        self.scaling = self.alpha/self.r
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        
    def forward(self, x):
        result = (x @ self.lora_a.transpose(0, 1) @ self.lora_b.transpose(0, 1)) * self.scaling
        return result



class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int, adapter: str = "", r: int = 4, 
                 enc_dec: Optional[str]=None, sampling: bool = False, sampling_loc: Optional[str]=None):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

        self.isLora = 'lora' in adapter # only apply if adapter is lora

        if self.isLora:
            # self.lora_query = LoRA(n_state, r=r)
            self.lora_key = LoRA(n_state, r=r)
            self.lora_value = LoRA(n_state, r=r)
            self.lora_out = LoRA(n_state, r=r)
        
        self.sampling = ('sampling' in adapter) and 'attweight' in adapter
        # if self.sampling and 'attweight' in adapter:
        #     self.cov_adapter = Linear(n_state*n_state)
        #     self.mean = nn.Parameter(torch.zeros(n_state))
        #     self.log_variance = nn.Parameter(torch.zeros(n_state))
        #     self.scale = nn.Parameter(torch.Tensor([0.01]))

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        if self.isLora:
            # if self.sampling:
            #     pass
            q = self.query(x)

            if kv_cache is None or xa is None or self.key not in kv_cache:
                # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
                # otherwise, perform key/value projections for self- or cross-attention as usual.
                k = self.key(x if xa is None else xa) + self.lora_key(x if xa is None else xa)
                v = self.value(x if xa is None else xa) + self.lora_value(x if xa is None else xa)
            else:
                # for cross-attention, calculate keys and values once and reuse in subsequent calls.
                k = kv_cache[self.key]
                v = kv_cache[self.value]

            wv, qk = self.qkv_attention(q, k, v, mask)

            out = self.out(wv) + self.lora_out(wv)
            return out, qk
        
        else:
            q = self.query(x)

            if kv_cache is None or xa is None or self.key not in kv_cache:
                # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
                # otherwise, perform key/value projections for self- or cross-attention as usual.
                k = self.key(x if xa is None else xa)
                v = self.value(x if xa is None else xa)
            else:
                # for cross-attention, calculate keys and values once and reuse in subsequent calls.
                k = kv_cache[self.key]
                v = kv_cache[self.value]

            wv, qk = self.qkv_attention(q, k, v, mask)
            return self.out(wv), qk

    def qkv_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        # return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), w


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False, 
                 adapter: str = "", 
                 r: int = 4,
                 enc_dec: Optional[str] = None,
                 sampling: bool = False,
                 sampling_loc: Optional[str] = None,
                 index: Optional[int] = None,
                 ):
        super().__init__()

        self.adapter = adapter
        self.sampling = sampling
        layer_for_sampling = 1
        
        # if index is specified, then it is encoder part
        if index is not None:
            # if index is even, apply sampling
            if index % layer_for_sampling == 0:
                self.apply_sampling = True
            # if odd, then no adapter added (only lora)
            else:
                self.apply_sampling = False
        else:
            self.apply_sampling = None

        if 'adapter' in self.adapter and not self.sampling:
            self.adapter_attn = Adapter(n_state)
            self.adapter_attn_ln = LayerNorm(n_state)

            self.adapter_mlp = Adapter(n_state)
            self.adapter_mlp_ln = LayerNorm(n_state)

        # elif 'lora_one' in self.adapter:
        #     self.adapter_one = LoRA(n_state)

        if "sampling" in self.adapter and self.sampling and self.apply_sampling:
            self.cov_adapter = AdapterReLU(n_state)
            # self.mean_adapter = AdapterSingle(n_state)
            # self.var_adapter = AdapterSingle(n_state, initialization="zero")

            
        # elif 'lora' in self.adapter:
        #     self.adapter_attn = LoRA(n_state)
        #     self.adapter_mlp = LoRA(n_state)

        self.attn = MultiHeadAttention(n_state, n_head, self.adapter, r, enc_dec, sampling, sampling_loc)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head, self.adapter, r, enc_dec, sampling, sampling_loc) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(
            Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
        kl_beta: float = 0.0,
    ):
        
        if self.adapter == 'adapter-lora_sampling_attout' and self.sampling and self.apply_sampling:
            attn_output = self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
            x = x + attn_output[0]

            if self.cross_attn:
                x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]

            # method 1
            x_before = x + self.mlp(self.mlp_ln(x))

            # ---------------- comment from here -------------------
            # cov = self.cov_adapter(self.mlp_ln(x))
            cov = self.cov_adapter(x_before)
            # scale = 0.1
            if self.training:
                N = 1
            else:
                N = 10
            x_after = torch.zeros(x.shape).to(x.device)
            for _ in range(N):
                normal_gaussian_noise = torch.randn_like(x_before)
                x_after += (x_before + (normal_gaussian_noise * cov))
                # x_after += (x_before - (normal_gaussian_noise * cov))
            x_after = x_after/N
            x = x_after
            # ------------------------------------------------------------

            # coba2
            # cov = self.var_adapter(x_before)
            # x_before = self.mean_adapter(x_before)

            if self.sampling and kl_beta != 0.0:
                # -------------------------- noise prior ---------------------------------
                noise = torch.randn_like(x)
                x_noise = x + noise
                # x_mask = torch.bernoulli(0.6 * torch.ones(x.shape)).to(x.device)
                # x_noise = x * x_mask
                attn_output_noise = self.attn(self.attn_ln(x_noise), mask=mask, kv_cache=kv_cache)
                x_noise = x_noise + attn_output_noise[0]

                if self.cross_attn:
                    x_noise = x_noise + self.cross_attn(self.cross_attn_ln(x_noise), xa, kv_cache=kv_cache)[0]

                x_noise_before = x_noise + self.mlp(self.mlp_ln(x_noise))
                # -------------------------- comment here ----------------------------------
                # method 1
                # cov = self.cov_adapter(self.mlp_ln(x))
                cov_noise = self.cov_adapter(x_noise_before)
                # scale = 0.1
                if self.training:
                    N = 1
                else:
                    N = 10
                x_noise_after = torch.zeros(x_noise.shape).to(x_noise.device)
                for _ in range(N):
                    normal_gaussian_noise = torch.randn_like(x_noise_before)
                    x_noise_after += (x_noise_before + (normal_gaussian_noise * cov_noise))
                    # x_noise_after += (x_noise_before - (normal_gaussian_noise * cov_noise))
                x_noise_after = x_noise_after/N
                x_noise = x_noise_after

                # --------------- normal gaussian prior -----------------------
                # x_noise_before = torch.zeros(x.shape).to(x.device)
                # cov_noise = torch.ones(x.shape).to(x.device)

                # ----------------- parameter prior --------------------------
                # batch_size = x.shape[0]
                # len_size = x.shape[1]
                # mu_p = self.p_mean.view(1, -1).expand(batch_size, len_size, -1)
                # std_p = torch.nn.functional.softplus(self.p_var.view(1, -1).expand(batch_size, len_size, -1))
                # x_noise_before = mu_p
                # cov_noise = std_p
                # --------------------------------------------------------------------------------------
                    
                # coba2
                # cov_noise = self.var_adapter(x_noise_before)
                # x_noise_before = self.mean_adapter(x_noise_before)

        elif self.apply_sampling == False:
            attn_output = self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
            x = x + attn_output[0]
            if self.cross_attn:
                x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
            x = x + self.mlp(self.mlp_ln(x))

        elif 'adapter' in self.adapter:
            attn_output = self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
            x = x + attn_output[0]
            x = self.adapter_attn(x)
            x = self.adapter_attn_ln(x)

            if self.cross_attn:
                x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]

            x = x + self.mlp(self.mlp_ln(x))
            x = self.adapter_mlp(x)
            x = self.adapter_mlp_ln(x)

        elif self.adapter == 'lora_one_sampling':
            attn_output = self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
            cov = self.cov_adapter(self.attn_ln(x))
            x = x + attn_output[0]

            if self.cross_attn:
                x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]

            
            # method 1
            x_before = x + self.mlp(self.mlp_ln(x))
            # scale = 0.1
            if self.training:
                N = 1
            else:
                N = 10
            x_after = torch.zeros(x.shape).to(x.device)
            for _ in range(N):
                normal_gaussian_noise = torch.randn_like(x_before)
                # print(x_after.device)
                # print(x_before.device)
                # print(normal_gaussian_noise.device)
                # print(cov.device)
                x_after += x_before + normal_gaussian_noise * cov * self.scale
            x_after = x_after/N
            x = x_after

            # elif self.adapter == 'lora':
            #     attn_output = self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
            #     lora = self.adapter_attn(self.attn_ln(x))
            #     x = x + attn_output[0] + lora

            #     if self.cross_attn:
            #         x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]

            #     lora = self.adapter_mlp(self.mlp_ln(x))
            #     x = x + self.mlp(self.mlp_ln(x))
            #     x = x + lora

            # elif self.adapter == 'lora_one':
            #     attn_output = self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
            #     lora = self.adapter_one(self.attn_ln(x))
            #     x = x + attn_output[0]

            #     if self.cross_attn:
            #         x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]

            #     x = x + self.mlp(self.mlp_ln(x)) + lora

        else:   # ------------- default -------------------
            attn_output = self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
            x = x + attn_output[0]
            if self.cross_attn:
                x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
            x = x + self.mlp(self.mlp_ln(x))

        if self.sampling and kl_beta != 0. and self.apply_sampling:
            return x, attn_output[1], (x_noise_before, cov_noise), (x_before, cov)
            # return x, attn_output[1]

        else:
            return x, attn_output[1], None, None
            # return x, attn_output[1]


class AudioEncoder(nn.Module):
    def __init__(
        self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, 
        adapter: str = "", r: int = 4,
        sampling:bool = False, sampling_loc: Optional[str] = None,
        lang_rescore: Optional[str] = 'none',
    ):
        super().__init__()
        # if lang_rescore == 'vib':
        #     self.mean_adapter = nn.Linear(n_state, n_state)
        #     self.var_adapter = nn.Linear(n_state, n_state)
        #     self.vib_ln = LayerNorm(n_state)

        #     self.mu_p = nn.Parameter(torch.randn(n_state))
        #     self.std_p = nn.Parameter(torch.randn(n_state))

        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, adapter=adapter, r=r, enc_dec="encoder", sampling=sampling, sampling_loc=sampling_loc, index=i+1) for i in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x, _ = block(x)

        x = self.ln_post(x)
        return x


class TextDecoder(nn.Module):
    def __init__(
        self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int,
        adapter: str = "", r: int = 4,
        sampling:bool = False, sampling_loc: Optional[str] = None,
        lang_rescore: str = "none",
    ):
        super().__init__()

        if "adapter" in lang_rescore:
            self.language_adapter = LanguageAdapter(n_state)
            # self.lang_ln = LayerNorm(n_state)
        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, cross_attention=True, adapter=adapter, r=r, enc_dec="decoder")
                for _ in range(n_layer)
            ]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
        x = (
            self.token_embedding(x)
            + self.positional_embedding[offset : offset + x.shape[-1]]
        )
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()

        return logits


class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions, 
                 adapter:str = "", 
                 r:int = 4, 
                 sampling:bool = False, 
                 sampling_loc: Optional[str] = None,
                 lang_rescore: str =  'none',
                 ):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            adapter=adapter,
            r=r,
            sampling=sampling,
            sampling_loc=sampling_loc,
            lang_rescore=lang_rescore,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            adapter=adapter,
            r=r,
            sampling=sampling,
            sampling_loc=sampling_loc,
            lang_rescore=lang_rescore
        )
        # use the last half among the decoder layers for time alignment by default;
        # to use a specific set of heads, see `set_alignment_heads()` below.
        all_heads = torch.zeros(
            self.dims.n_text_layer, self.dims.n_text_head, dtype=torch.bool
        )
        all_heads[self.dims.n_text_layer // 2 :] = True
        self.register_buffer("alignment_heads", all_heads.to_sparse(), persistent=False)

    def set_alignment_heads(self, dump: bytes):
        array = np.frombuffer(
            gzip.decompress(base64.b85decode(dump)), dtype=bool
        ).copy()
        mask = torch.from_numpy(array).reshape(
            self.dims.n_text_layer, self.dims.n_text_head
        )
        self.register_buffer("alignment_heads", mask.to_sparse(), persistent=False)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(
        self, mel: torch.Tensor, tokens: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab >= 51865

    @property
    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual)

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = decode_function
