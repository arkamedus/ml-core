import importlib
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import math

import torch
from tqdm import tqdm
import io
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from itertools import islice


try:
    from datasets import load_dataset, Audio as HFAudio
    _HAS_HF = True
except Exception:
    _HAS_HF = False
try:
    import soundfile as sf
    _HAS_SF = True
except Exception:
    sf = None
    _HAS_SF = False
try:
    import audioread
    _HAS_AR = True
except Exception:
    audioread = None
    _HAS_AR = False
try:
    import torchaudio
    _HAS_TA = True
except Exception:
    torchaudio = None
    _HAS_TA = False


def get(sample, k):
    return None if not k else sample.get(k, None)

def _mono(w):
    if w.ndim == 2 and w.shape[0] > 1:
        w = w.mean(dim=0, keepdim=True)
    if w.ndim == 1:
        w = w.unsqueeze(0)
    return w


def _decode_bytes_soundfile(b):
    if not _HAS_SF:
        return None
    data, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)  # (T,C)
    wav = torch.from_numpy(data.T)  # (C,T)
    return wav, int(sr)


def _decode_bytes_audioread(b):
    if not _HAS_AR:
        return None
    with audioread.audio_open(io.BytesIO(b)) as f:
        sr = int(f.samplerate)
        ch = int(f.channels)
        chunks = []
        for buf in f:
            x = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
            chunks.append(x)
        if not chunks:
            return None
        x = np.concatenate(chunks, axis=0)
        if ch > 1:
            x = x.reshape(-1, ch).T  # (C,T)
        else:
            x = x.reshape(1, -1)
        wav = torch.from_numpy(x)
        return wav, sr


def _resample_1d(wav_1d, sr_in, sr_out):
    if sr_in == sr_out:
        return wav_1d
    if _HAS_TA:
        try:
            return torchaudio.functional.resample(wav_1d.unsqueeze(0), int(sr_in), int(sr_out)).squeeze(0)
        except Exception:
            pass
    x = wav_1d.unsqueeze(0).unsqueeze(0)  # (1,1,T)
    T = x.size(-1)
    T2 = max(1, int(round(T * float(sr_out) / float(sr_in))))
    return torch.nn.functional.interpolate(x, size=T2, mode="linear", align_corners=False).squeeze(0).squeeze(0)


def to_audio_wave(x, sr_in=16000, sr_out=16000):
    if x is None:
        return None

    if isinstance(x, dict):
        arr = x.get("array", None)
        if arr is not None:
            wav = torch.tensor(arr, dtype=torch.float32)
            sr = int(x.get("sampling_rate", sr_in))
            return _mono(wav), sr

        b = x.get("bytes", None)
        if isinstance(b, (bytes, bytearray)) and len(b) > 0:
            out = _decode_bytes_soundfile(b)
            if out is None:
                out = _decode_bytes_audioread(b)
            if out is None:
                return None
            wav, sr = out
            return _mono(wav.float()), int(sr)

        return None

    if isinstance(x, np.ndarray):
        wav = torch.tensor(x, dtype=torch.float32)
        return _mono(wav), sr_out

    if torch.is_tensor(x):
        return _mono(x.float()), sr_out

    return None


@dataclass
class DatasetSpec:
    source: Any
    split: str = "train"
    name: str = "ds"
    field_map: Dict[str, str] = None
    load_kwargs: Dict[str, Any] = None
    fallback_streaming: bool = False
    streaming_take: int = 5000
    max_samples: Optional[int] = None


class MultiModalAudioText(Dataset):
    def __init__(self, specs):
        self.datasets = []
        self.names = []

        if not _HAS_HF:
            for s in specs:
                self.datasets.append((s.source, s.field_map or {}))
                self.names.append(s.name)
        else:
            for s in specs:
                fm = s.field_map or {}
                if isinstance(s.source, str):
                    kw = dict(s.load_kwargs or {})
                    try:
                        ds = load_dataset(s.source, split=s.split, **kw)
                        audio_col = fm.get("audio")
                        if audio_col and hasattr(ds, "cast_column") and audio_col in ds.column_names:
                            ds = ds.cast_column(audio_col, HFAudio(decode=False))
                    except Exception as e:
                        if not s.fallback_streaming:
                            raise
                        kw2 = dict(kw)
                        kw2["streaming"] = True
                        it = load_dataset(s.source, split=s.split, **kw2)
                        ds = list(islice(iter(it), s.streaming_take))
                        print(f"[{s.name}] streaming fallback | {type(e).__name__}: {e}")
                else:
                    ds = s.source
                    audio_col = fm.get("audio")
                    if audio_col and hasattr(ds, "cast_column") and hasattr(ds, "column_names") and audio_col in ds.column_names:
                        ds = ds.cast_column(audio_col, HFAudio(decode=False))

                if s.max_samples is not None:
                    ds = ds.select(range(min(len(ds), s.max_samples))) if hasattr(ds, "select") else list(ds)[:s.max_samples]

                self.datasets.append((ds, fm))
                self.names.append(s.name)

        self.index = []
        for di, (ds, _) in enumerate(self.datasets):
            self.index.extend([(di, i) for i in range(len(ds))])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        di, si = self.index[idx]
        ds, fm = self.datasets[di]
        row = ds[si]
        text = str(get(row, fm.get("text"))).upper()
        audio = to_audio_wave(get(row, fm.get("audio")))
        return {"text": text, "audio": audio, "source": self.names[di]}


def _voice_from_wav(wav_1d, sr, steps=16, f_lo=300.0, f_hi=3500.0, n_fft=1024, hop=256, tilt_alpha=0.2, frame_norm=True):
    if wav_1d is None or wav_1d.numel() == 0:
        acc = torch.zeros(steps)
        edges = torch.linspace(0.0, float(sr) * 0.5, steps + 1)
        top2 = torch.tensor([10, 10], dtype=torch.long).clamp(0, steps - 1)
        freqs = torch.fft.rfftfreq(n_fft, d=1.0 / float(sr))
        mags = torch.zeros_like(freqs)
        return acc, edges, top2, f"{int(top2[0]):01x}{int(top2[1]):01x}", freqs, mags

    x = wav_1d.float() - wav_1d.float().mean()
    win = torch.hann_window(n_fft, device=x.device)

    spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=win, return_complex=True)
    mag_tf = spec.abs()

    if frame_norm:
        denom = mag_tf.mean(dim=0, keepdim=True).clamp_min(1e-6)
        mag_tf = mag_tf / denom

    mag = mag_tf.mean(dim=1)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / float(sr)).to(x.device)

    m = (freqs >= f_lo) & (freqs <= f_hi)
    freqs_b = freqs[m]
    mag_b = torch.log(mag[m] + 1e-8)

    if tilt_alpha:
        f_ref = torch.tensor(float(f_lo), device=x.device, dtype=freqs_b.dtype)
        mag_b = mag_b + float(tilt_alpha) * torch.log((freqs_b / f_ref).clamp_min(1e-6))

    edges = torch.logspace(
        torch.log10(torch.tensor(float(f_lo), device=x.device)),
        torch.log10(torch.tensor(float(f_hi), device=x.device)),
        steps + 1
    )

    bin_idx = torch.bucketize(freqs_b, edges, right=False) - 1
    bin_idx = bin_idx.clamp(0, steps - 1)

    acc = torch.zeros(steps, device=x.device, dtype=mag_b.dtype)
    acc.scatter_add_(0, bin_idx, mag_b)

    top2 = torch.topk(acc, k=2, largest=True).indices
    voice_hex = f"{int(top2[0]):01x}{int(top2[1]):01x}"
    return acc, edges, top2, voice_hex, freqs_b, mag_b

def trim_silence(
        wav: torch.Tensor,
        sr: int,
        rel_thresh: float = 0.025,  # 10% of peak amplitude
        min_silence_ms: float = 20.0,  # smooth envelope window
        keep_ms: float = 20.0,  # keep a little context around speech
        min_len_ms: float = 80.0,  # if too short after trim, don't trim
):
    """
    Trim leading/trailing silence from wav.

    Args:
      wav: Tensor shape (T,) or (1,T) or (C,T). Float in [-1,1] ideally.
      sr: sampling rate
      rel_thresh: threshold as fraction of max abs amplitude (0.10 = 10%)
      min_silence_ms: envelope smoothing window in ms (avoid chopping on transients)
      keep_ms: keep this many ms before/after detected region
      min_len_ms: if trimmed audio would be shorter than this, return original

    Returns:
      wav_trim: Tensor (T,) (mono) on same device
      (start, end): sample indices used from original (end exclusive)
    """
    if wav is None:
        return wav, (0, 0)

    # collapse to mono (T,)
    if wav.ndim == 2:
        if wav.size(0) > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav.squeeze(0)
    elif wav.ndim == 1:
        pass
    else:
        # weird shape, best effort
        wav = wav.reshape(-1)

    T = wav.numel()
    if T == 0:
        return wav, (0, 0)

    peak = wav.abs().max()
    if peak <= 1e-8:
        # all silence
        return wav, (0, T)

    thr = peak * float(rel_thresh)

    # smoothed abs envelope (moving average)
    win = max(1, int(sr * (min_silence_ms / 1000.0)))
    if win > 1:
        env = F.avg_pool1d(wav.abs().view(1, 1, -1), kernel_size=win, stride=1, padding=win // 2).view(-1)
    else:
        env = wav.abs()

    active = env > thr
    idx = torch.nonzero(active, as_tuple=False).view(-1)
    if idx.numel() == 0:
        # nothing above threshold
        return wav, (0, T)

    start = int(idx[0].item())
    end = int(idx[-1].item()) + 1

    # keep margins
    keep = int(sr * (keep_ms / 1000.0))
    start = max(0, start - keep)
    end = min(T, end + keep)

    # don't over-trim into useless tiny clips
    min_len = int(sr * (min_len_ms / 1000.0))
    if (end - start) < min_len:
        return wav, (0, T)

    return wav[start:end], (start, end)


def make_collate_audio_text(vocab, max_text_len=128, audio_sr=16000, audio_len=1, trim=True):

    _DEFAULT_VOICE_HEX = "aa"
    _VOICE_STEPS = 16

    def _voice_ids_from_hex(h):
        h = (h or _DEFAULT_VOICE_HEX).strip().lower()
        if len(h) != 2:
            h = _DEFAULT_VOICE_HEX
        try:
            return int(h[0], 16), int(h[1], 16)
        except Exception:
            return int(_DEFAULT_VOICE_HEX[0], 16), int(_DEFAULT_VOICE_HEX[1], 16)

    def collate(batch):
        B = len(batch)

        ids_list, has_text = [], []
        for b in batch:
            t = b.get("text", "")
            if isinstance(t, str) and len(t) > 0:
                ids = vocab.tokenize(t)

                # reserve room for BOS + EOS
                ids = ids[: max_text_len - 2]

                ids = [vocab.BOS_ID] + ids
                if len(ids) == 1 or ids[-1] != vocab.EOS_ID:
                    ids = ids + [vocab.EOS_ID]

                ids = ids[:max_text_len]
                ids_list.append(torch.tensor(ids, dtype=torch.long))
                has_text.append(True)
            else:
                # If no text, still create a tiny valid sequence: BOS EOS
                ids_list.append(torch.tensor([vocab.BOS_ID, vocab.EOS_ID], dtype=torch.long))
                has_text.append(False)

        L = max([x.numel() for x in ids_list] + [1])
        input_ids = torch.full((B, L), vocab.PAD_ID, dtype=torch.long)
        attn_mask = torch.zeros((B, L), dtype=torch.bool)

        for i, x in enumerate(ids_list):
            if x.numel() > 0:
                input_ids[i, :x.numel()] = x
                attn_mask[i, :x.numel()] = True

        wav_list, wav_len_list, has_audio = [], [], []
        voice_ids_list, voice_hex_list = [], []

        for b in batch:
            a = b.get("audio")
            if a is None:
                wav_list.append(torch.zeros(1))
                wav_len_list.append(0)
                has_audio.append(False)
                va, vb = _voice_ids_from_hex(_DEFAULT_VOICE_HEX)
                voice_ids_list.append((va, vb))
                voice_hex_list.append(_DEFAULT_VOICE_HEX)
                continue

            wav, sr = a
            wav = wav.float()

            if wav.ndim == 2:
                wav = wav.mean(dim=0)
            elif wav.ndim != 1:
                wav = wav.reshape(-1)

            if int(sr) != int(audio_sr):
                wav = _resample_1d(wav, int(sr), int(audio_sr))

            if trim:
                wav, _ = trim_silence(wav, audio_sr, min_silence_ms=20.0, keep_ms=10.0, min_len_ms=80.0)

            max_T = int(float(audio_sr) * float(audio_len))
            if max_T > 0 and wav.numel() > max_T:
                wav = wav[:max_T]

            wav = wav / wav.abs().max().clamp_min(1e-6)

            acc, edges, top2, vh, freqs, mags = _voice_from_wav(wav, audio_sr, steps=_VOICE_STEPS)

            voice_ids_list.append((int(top2[0]), int(top2[1])))
            voice_hex_list.append(vh)

            wav_list.append(wav)
            wav_len_list.append(int(wav.numel()))
            has_audio.append(True)

        Tmax = max([w.numel() for w in wav_list] + [1])
        wav = torch.stack([F.pad(w, (0, Tmax - w.numel())) for w in wav_list], dim=0)
        wav_len = torch.tensor(wav_len_list, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attn_mask": attn_mask,
            "wav": wav,
            "wav_len": wav_len,
            "has_text": torch.tensor(has_text, dtype=torch.bool),
            "has_audio": torch.tensor(has_audio, dtype=torch.bool),
            "audio_sr": int(audio_sr),
            "voice_ids": torch.tensor(voice_ids_list, dtype=torch.long),
            "voice_hex": voice_hex_list,
        }

    return collate


def init(m):
    # ----- Linear -----
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # ----- Conv / ConvTranspose -----
    elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        # Kaiming init must match the activation actually used in these stacks.
        # If you use ReLU (see §2), this is correct.
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    # ----- Embedding -----
    elif isinstance(m, nn.Embedding):
        # small-variance controlled init
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        # keep padding row stable if padding_idx is used
        if getattr(m, "padding_idx", None) is not None:
            with torch.no_grad():
                m.weight[m.padding_idx].fill_(0.0)

    # ----- LayerNorm -----
    elif isinstance(m, nn.LayerNorm):
        if m.elementwise_affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

def _pow2_strides_for_hop(hop: int):
    # smallest dynamic rule: hop must be power of 2
    if hop <= 0 or (hop & (hop - 1)) != 0:
        raise ValueError(f"hop must be a power of 2 (e.g., 64/128/256). Got {hop}.")
    return [2] * int(math.log2(hop))

class ConvEncoder(nn.Module):
    # wav: (B,T) -> z: (B, D, T//hop)
    def __init__(self, D=128, hop=256, drop=0.0):
        super().__init__()
        # hop=256 => 8 stages stride 2
        strides = _pow2_strides_for_hop(hop)
        assert math.prod(strides) == hop
        ch = D
        layers = []
        in_ch = 1
        for s in strides:
            layers += [
                nn.Conv1d(in_ch, ch, kernel_size=7, stride=s, padding=3),
                nn.GELU(),
                nn.Dropout(drop),
            ]
            in_ch = ch
        self.net = nn.Sequential(*layers)

    def forward(self, wav):
        x = wav.unsqueeze(1)  # (B,1,T)
        return self.net(x)  # (B,D,Tf)

class ConvDecoder(nn.Module):
    # zq: (B, D, Tf) -> wav_hat: (B,T)
    def __init__(self, D=128, hop=256, drop=0.0):
        super().__init__()
        strides = _pow2_strides_for_hop(hop)
        assert math.prod(strides) == hop
        ch = D
        layers = []
        in_ch = D
        for s in strides:
            layers += [
                nn.ConvTranspose1d(in_ch, ch, kernel_size=4, stride=s, padding=1),
                nn.GELU(),
                nn.Dropout(drop),
            ]
            in_ch = ch
        layers += [nn.Conv1d(ch, 1, kernel_size=7, padding=3)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        y = self.net(z)  # (B,1,T)
        return torch.tanh(y.squeeze(1))

class VectorQuantizer(nn.Module):
    """
    Straight-through VQ + OPTIONAL differentiable usage regularizer.

    Inputs:  z (B,D,T)
    Outputs: z_q (B,D,T), codes (B,T), loss (scalar)
    """
    def __init__(
        self,
        n_codes=512,
        D=128,
        beta=0.25,
        usage_weight: float = 0.0,   # set >0 to enable
        usage_temp: float = 0.5,     # softmax temperature
        usage_mode: str = "entropy", # "entropy" or "kl_uniform"
    ):
        super().__init__()
        self.n_codes = n_codes
        self.D = D
        self.beta = beta
        self.usage_weight = float(usage_weight)
        self.usage_temp = float(usage_temp)
        self.usage_mode = usage_mode

        self.codebook = nn.Embedding(n_codes, D)
        nn.init.uniform_(self.codebook.weight, -1.0 / n_codes, 1.0 / n_codes)

    def forward(self, z):
        B, D, T = z.shape
        z_t = z.permute(0, 2, 1).contiguous()
        flat = z_t.view(B*T, D)

        cb = self.codebook.weight

        dist = (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ cb.t()
            + cb.pow(2).sum(1).unsqueeze(0)
        )  # (BT, N)

        codes = torch.argmin(dist, dim=1)
        z_q = self.codebook(codes).view(B, T, D)

        z_q_det = z_q.detach()
        z_det = z_t.detach()
        commit = F.mse_loss(z_t, z_q_det)
        codebk = F.mse_loss(z_q, z_det)
        loss = codebk + self.beta * commit

        z_q = z_t + (z_q - z_t).detach()
        z_q = z_q.permute(0, 2, 1).contiguous()
        codes = codes.view(B, T)

        # RETURN DISTANCES TOO
        return z_q, codes, loss, dist.view(B, T, self.n_codes)

class ResidualVQ(nn.Module):
    def __init__(
        self,
        K=4,
        n_codes=512,
        D=128,
        beta=0.25,
        usage_weight: float = 0.0,
        usage_temp: float = 0.5,
        usage_mode: str = "entropy",
    ):
        super().__init__()
        self.K = K
        self.vqs = nn.ModuleList([
            VectorQuantizer(
                n_codes=n_codes,
                D=D,
                beta=beta,
                usage_weight=usage_weight,
                usage_temp=usage_temp,
                usage_mode=usage_mode,
            )
            for _ in range(K)
        ])

    def forward(self, z):
        residual = z
        z_sum = torch.zeros_like(z)
        all_codes = []
        all_dist = []
        total_loss = 0.0

        for vq in self.vqs:
            z_q, codes, loss, dist = vq(residual)     # NOW 4
            z_sum = z_sum + z_q
            residual = residual - z_q.detach()
            all_codes.append(codes.unsqueeze(1))      # (B,1,T)
            all_dist.append(dist.unsqueeze(1))        # (B,1,T,N)
            total_loss = total_loss + loss

        codes = torch.cat(all_codes, dim=1)          # (B,K,T)
        dist = torch.cat(all_dist, dim=1)            # (B,K,T,N)
        return z_sum, codes, total_loss, dist

class AudioCodec(nn.Module):
    def __init__(
        self,
        D=128,
        hop=256,
        K=4,
        n_codes=512,
        beta=0.25,
        drop=0.0,
        usage_weight: float = 0.0,
        usage_temp: float = 0.5,
        usage_mode: str = "entropy",
    ):
        super().__init__()
        self.D = D
        self.hop = hop
        self.K = K
        self.n_codes = n_codes

        self.enc = ConvEncoder(D=D, hop=hop, drop=drop)
        self.rvq = ResidualVQ(
            K=K, n_codes=n_codes, D=D, beta=beta,
            usage_weight=usage_weight,
            usage_temp=usage_temp,
            usage_mode=usage_mode,
        )
        self.dec = ConvDecoder(D=D, hop=hop, drop=drop)
        self.apply(init)

    def forward(self, wav):
        z = self.enc(wav)
        z = torch.tanh(z)
        z_q, codes, qloss, dist = self.rvq(z)  # NOW 4
        wav_hat = self.dec(z_q)
        return wav_hat, codes, qloss, dist

    @torch.no_grad()
    def encode_codes(self, wav):
        z = self.enc(wav)
        z = torch.tanh(z)
        _, codes, _, _ = self.rvq(z)
        return codes

    @torch.no_grad()
    def decode_codes(self, codes):
        B, K, Tf = codes.shape
        assert K == self.K
        z_sum = torch.zeros(B, self.D, Tf, device=codes.device)
        for k, vq in enumerate(self.rvq.vqs):
            emb = vq.codebook(codes[:, k, :])          # (B,Tf,D)
            emb = emb.permute(0, 2, 1).contiguous()    # (B,D,Tf)
            z_sum = z_sum + emb
        wav_hat = self.dec(z_sum)
        return wav_hat