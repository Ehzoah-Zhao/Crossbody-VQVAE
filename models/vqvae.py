import torch
import torch.nn as nn
from models.encdec import Encoder, Decoder
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset


class CrossEmbodimentVQVAE(nn.Module):
    """
    双编码器 + 共享码本 + 双解码器 的跨本体 VQ-VAE

    数据流：
        Encoder_A (input_dim_A) ──┐
                                   ├── 共享 VQ 码本 ──┬── Decoder_A (→ input_dim_A)
        Encoder_B (input_dim_B) ──┘                  └── Decoder_B (→ input_dim_B)

    自重建:   forward_A / forward_B
    跨本体:   forward_AB (A→B̂),  forward_BA (B→Â)
    循环路径: cycle_ABA (A→B̂→Â), cycle_BAB (B→Â→B̂)
    """

    def __init__(
        self,
        input_dim_A: int,           # SMPL: 134
        input_dim_B: int,           # G1:   186
        nb_code: int        = 512,
        code_dim: int       = 512,
        output_emb_width: int = 512,
        down_t: int         = 3,
        stride_t: int       = 2,
        width: int          = 512,
        depth: int          = 3,
        dilation_growth_rate: int = 3,
        activation: str     = 'relu',
        norm                = None,
        quantizer_type: str = 'ema_reset',
        mu: float           = 0.99,
    ):
        super().__init__()
        self.code_dim = code_dim

        # ── 公共的 encoder/decoder kwargs ─────────────────────────
        _enc_kw = dict(
            output_emb_width  = output_emb_width,
            down_t            = down_t,
            stride_t          = stride_t,
            width             = width,
            depth             = depth,
            dilation_growth_rate = dilation_growth_rate,
            activation        = activation,
            norm              = norm,
        )

        # ── 两个独立 Encoder ──────────────────────────────────────
        # encdec.py 的第一个参数 input_emb_width 就是输入特征维度，直接传入即可
        self.encoder_A = Encoder(input_dim_A, **_enc_kw)
        self.encoder_B = Encoder(input_dim_B, **_enc_kw)

        # ── 共享 VQ 码本 ──────────────────────────────────────────
        # QuantizeEMAReset 需要 args.mu，用轻量 namespace 传入
        class _Args:
            pass
        _a = _Args()
        _a.mu = mu

        _quant_cls = {
            'ema_reset': QuantizeEMAReset,
            'ema':       QuantizeEMA,
            'reset':     QuantizeReset,
        }
        if quantizer_type == 'orig':
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif quantizer_type in _quant_cls:
            self.quantizer = _quant_cls[quantizer_type](nb_code, code_dim, _a)
        else:
            raise ValueError(f"未知 quantizer 类型: {quantizer_type}")

        # ── 两个独立 Decoder ──────────────────────────────────────
        self.decoder_A = Decoder(input_dim_A, **_enc_kw)
        self.decoder_B = Decoder(input_dim_B, **_enc_kw)

    # ─────────────────────────────────────────────────────────────
    # 内部工具
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _pre(x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) → (B, C, T)  Conv1d 所需的通道优先格式"""
        return x.permute(0, 2, 1).float()

    @staticmethod
    def _post(x: torch.Tensor) -> torch.Tensor:
        """(B, C, T) → (B, T, C)"""
        return x.permute(0, 2, 1)

    def _encode_feat(self, x: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        """
        量化前的连续编码特征，shape: (B, T', C)
        T' = T / (stride_t ** down_t)，即时间下采样后的长度
        供量化和对比损失使用
        """
        return self._post(encoder(self._pre(x)))

    def _quantize(self, feat: torch.Tensor):
        """
        feat: (B, T', C)
        返回: x_q (B, C, T'),  commit_loss (scalar),  perplexity (scalar)
        VQ 内部使用 straight-through estimator，梯度可回传
        """
        # quantizer 期望 (B, C, T')
        x_q, loss, ppl = self.quantizer(feat.permute(0, 2, 1))
        return x_q, loss, ppl

    def _decode(self, x_q: torch.Tensor, decoder: nn.Module) -> torch.Tensor:
        """x_q: (B, C, T') → (B, T, input_dim)"""
        return self._post(decoder(x_q))

    # ─────────────────────────────────────────────────────────────
    # 自重建（训练基础）
    # ─────────────────────────────────────────────────────────────

    def forward_A(self, x_A: torch.Tensor):
        """A → Enc_A → VQ → Dec_A → Â   返回 (Â, commit_loss, perplexity)"""
        feat         = self._encode_feat(x_A, self.encoder_A)
        x_q, loss, ppl = self._quantize(feat)
        return self._decode(x_q, self.decoder_A), loss, ppl

    def forward_B(self, x_B: torch.Tensor):
        """B → Enc_B → VQ → Dec_B → B̂   返回 (B̂, commit_loss, perplexity)"""
        feat         = self._encode_feat(x_B, self.encoder_B)
        x_q, loss, ppl = self._quantize(feat)
        return self._decode(x_q, self.decoder_B), loss, ppl

    # ─────────────────────────────────────────────────────────────
    # 跨本体生成（核心目标）
    # ─────────────────────────────────────────────────────────────

    def forward_AB(self, x_A: torch.Tensor):
        """A → Enc_A → VQ → Dec_B → B̂   SMPL 动作 → G1 动作"""
        feat         = self._encode_feat(x_A, self.encoder_A)
        x_q, loss, ppl = self._quantize(feat)
        return self._decode(x_q, self.decoder_B), loss, ppl

    def forward_BA(self, x_B: torch.Tensor):
        """B → Enc_B → VQ → Dec_A → Â   G1 动作 → SMPL 动作"""
        feat         = self._encode_feat(x_B, self.encoder_B)
        x_q, loss, ppl = self._quantize(feat)
        return self._decode(x_q, self.decoder_A), loss, ppl

    # ─────────────────────────────────────────────────────────────
    # 循环一致性路径（两跳，直通梯度自动传播）
    # ─────────────────────────────────────────────────────────────

    def cycle_ABA(self, x_A: torch.Tensor) -> torch.Tensor:
        """
        A → B̂ → Â_cycle
        训练目标: Â_cycle ≈ x_A
        梯度路径: Dec_B → Enc_B → VQ(straight-through) → Dec_A
        """
        x_B_hat, _, _ = self.forward_AB(x_A)     # 连续张量，可微
        x_A_cycle, _, _ = self.forward_BA(x_B_hat)
        return x_A_cycle

    def cycle_BAB(self, x_B: torch.Tensor) -> torch.Tensor:
        """
        B → Â → B̂_cycle
        训练目标: B̂_cycle ≈ x_B
        """
        x_A_hat, _, _ = self.forward_BA(x_B)
        x_B_cycle, _, _ = self.forward_AB(x_A_hat)
        return x_B_cycle

    # ─────────────────────────────────────────────────────────────
    # 对比损失所需的全局动作嵌入
    # ─────────────────────────────────────────────────────────────

    def get_motion_emb_A(self, x_A: torch.Tensor) -> torch.Tensor:
        """均值池化量化前特征 → (B, C)  正样本对中 A 侧的表示"""
        return self._encode_feat(x_A, self.encoder_A).mean(dim=1)

    def get_motion_emb_B(self, x_B: torch.Tensor) -> torch.Tensor:
        """均值池化量化前特征 → (B, C)  正样本对中 B 侧的表示"""
        return self._encode_feat(x_B, self.encoder_B).mean(dim=1)

    # ─────────────────────────────────────────────────────────────
    # 推理：编码为离散 token 索引
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode_A(self, x_A: torch.Tensor) -> torch.Tensor:
        """返回 SMPL 动作的码本索引序列，shape: (B, T')"""
        N    = x_A.shape[0]
        feat = self._encode_feat(x_A, self.encoder_A)
        flat = feat.contiguous().view(-1, feat.shape[-1])
        return self.quantizer.quantize(flat).view(N, -1)

    @torch.no_grad()
    def encode_B(self, x_B: torch.Tensor) -> torch.Tensor:
        """返回 G1 动作的码本索引序列，shape: (B, T')"""
        N    = x_B.shape[0]
        feat = self._encode_feat(x_B, self.encoder_B)
        flat = feat.contiguous().view(-1, feat.shape[-1])
        return self.quantizer.quantize(flat).view(N, -1)