"""
UNet++ (UNet Plus Plus) Architecture
=====================================
Paper : "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
        Zhou et al., 2018  →  https://arxiv.org/abs/1807.10165

Key idea
--------
Standard UNet connects each encoder node directly to its mirrored decoder node
(a single skip connection per depth level).

UNet++ replaces those sparse, direct bridges with *dense nested sub-networks*.
Every intermediate node X[i][j]  (i = depth, j = dense-block step) is computed
from ALL previous nodes on the same depth row  PLUS  an upsampled node from
the level directly below.  This lets the network learn to close the semantic
gap between encoder and decoder features before they are merged.

Node formula
------------
    X[i][j] = H( [X[i][k] for k in 0..j-1]  +  Up(X[i+1][j-1]) )

where H is a DoubleConv block and + denotes concatenation along the channel dim.

Architecture shape (depth=4, base_filters=32)
----------------------------------------------
    depth  |  j=0 (encoder)  |  j=1   j=2   j=3   j=4 (decoder outputs)
    -------|-----------------|--------------------------------------
      0    |  X[0][0]         |  X[0][1]  X[0][2]  X[0][3]  X[0][4]
      1    |  X[1][0]         |  X[1][1]  X[1][2]  X[1][3]
      2    |  X[2][0]         |  X[2][1]  X[2][2]
      3    |  X[3][0]         |  X[3][1]
    -------|-----------------|--------------------------------------
    bottle |  X[4][0] (no skip connections above this)

Further reading
---------------
- Original paper  : https://arxiv.org/abs/1807.10165
- Follow-up paper : "UNet 3+: A Full-Scale Connected UNet" (2020)
                    https://arxiv.org/abs/2004.08790
- Blog (Lilian Weng's overview of medical segmentation)
                    https://lilianweng.github.io/posts/2017-10-29-object-recognition/
- Blog (Papers with Code walkthrough)
                    https://paperswithcode.com/method/unet
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building block  (identical to your UNet implementation)
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two consecutive  Conv → BN → ReLU  layers."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# UNet++
# ---------------------------------------------------------------------------

class UNetPP(nn.Module):
    """
    UNet++ with configurable depth and base channel width.

    Parameters
    ----------
    in_channels   : number of input image channels  (e.g. 3 for RGB)
    out_channels  : number of segmentation classes   (e.g. 1 for binary mask)
    depth         : number of pooling stages         (paper uses 4)
    base_filters  : channel count at the first encoder stage;
                    doubles at each deeper level     (paper uses 32 or 64)
    deep_supervision : if True, produce one output map per decoder column
                       and average them during training (paper Section 3.2).
                       At inference the final (rightmost) head is used alone.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        depth: int = 4,
        base_filters: int = 32,
        deep_supervision: bool = False,
    ):
        super().__init__()

        self.depth = depth
        self.deep_supervision = deep_supervision

        # ------------------------------------------------------------------
        # Channel widths at each depth level
        #   filters[0] = base_filters  (shallowest)
        #   filters[d] = base_filters * 2**d  (deepest / bottleneck)
        # ------------------------------------------------------------------
        self.filters = [base_filters * (2 ** d) for d in range(depth + 1)]

        # ------------------------------------------------------------------
        # Pool and upsample helpers  (shared across all nodes)
        # ------------------------------------------------------------------
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # One ConvTranspose per depth level: from filters[i+1] → filters[i]
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(self.filters[i + 1], self.filters[i], kernel_size=2, stride=2)
            for i in range(depth)
        ])

        # ------------------------------------------------------------------
        # Node convolutions  X[i][j]
        #
        # X[i][0]  →  encoder nodes  (just a DoubleConv on the pooled input)
        # X[i][j]  →  dense nodes    (j >= 1)
        #
        # Input channels for X[i][j] with j >= 1:
        #   • (j) previous same-row outputs, each with filters[i] channels
        #   • 1 upsampled node from depth i+1, with filters[i] channels
        #   → total = (j + 1) * filters[i]
        # ------------------------------------------------------------------

        # encoder_nodes[i] = X[i][0]   for i in 0..depth
        self.encoder_nodes = nn.ModuleList()
        prev_ch = in_channels
        for i in range(depth + 1):
            self.encoder_nodes.append(DoubleConv(prev_ch, self.filters[i]))
            prev_ch = self.filters[i]

        # dense_nodes[i][j-1] = X[i][j]  for i in 0..depth-1, j in 1..depth-i
        # We store as a 2-D ModuleList (list of lists).
        self.dense_nodes = nn.ModuleList()
        for i in range(depth):                        # depth row
            row = nn.ModuleList()
            for j in range(1, depth - i + 1):        # dense column
                in_ch = (j + 1) * self.filters[i]    # see formula above
                row.append(DoubleConv(in_ch, self.filters[i]))
            self.dense_nodes.append(row)

        # ------------------------------------------------------------------
        # Output heads
        # With deep_supervision we have one head per decoder column (j=1..depth).
        # Without it we only need the final column head (j=depth).
        # ------------------------------------------------------------------
        n_heads = depth if deep_supervision else 1
        self.output_heads = nn.ModuleList([
            nn.Conv2d(self.filters[0], out_channels, kernel_size=1)
            for _ in range(n_heads)
        ])

    # -----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_channels, H, W)
        returns : (B, out_channels, H, W)
        """

        # ------------------------------------------------------------------
        # node_outputs[i][j]  stores the feature map at grid position (i, j).
        # We build this column by column (j = 0 first, then j = 1, 2, …).
        # ------------------------------------------------------------------

        # Initialise with empty lists; we will fill them as we go.
        node_outputs = [[] for _ in range(self.depth + 1)]

        # ------ j = 0 : encoder column ------------------------------------
        #   X[0][0] = DoubleConv(input)
        #   X[i][0] = DoubleConv(Pool(X[i-1][0]))   for i >= 1
        current = x
        for i in range(self.depth + 1):
            feat = self.encoder_nodes[i](current)
            node_outputs[i].append(feat)             # node_outputs[i][0]
            if i < self.depth:
                current = self.pool(feat)            # pool for next depth

        # ------ j = 1 .. depth : dense columns ----------------------------
        #   X[i][j] = DoubleConv( cat(X[i][0..j-1],  Up(X[i+1][j-1])) )
        for j in range(1, self.depth + 1):
            for i in range(self.depth - j + 1):     # valid rows for this j

                # All previous same-row outputs  (j of them)
                same_row = node_outputs[i]           # list of j feature maps

                # Upsampled output from one level below
                up_feat = self.up[i](node_outputs[i + 1][j - 1])

                # Concatenate: same_row features + upsampled
                merged = torch.cat([*same_row, up_feat], dim=1)

                # Apply the dense conv node   (dense_nodes[i][j-1])
                feat = self.dense_nodes[i][j - 1](merged)
                node_outputs[i].append(feat)         # node_outputs[i][j]

        # ------------------------------------------------------------------
        # Output
        # ------------------------------------------------------------------
        if self.deep_supervision:
            # One output per decoder column, then average
            # Column j gives X[0][j] for j = 1 .. depth
            outputs = [
                self.output_heads[j - 1](node_outputs[0][j])
                for j in range(1, self.depth + 1)
            ]
            return torch.mean(torch.stack(outputs, dim=0), dim=0)
        else:
            # Single output from the final decoder column: X[0][depth]
            return self.output_heads[0](node_outputs[0][self.depth])


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Mimic a batch of 2 RGB images at 256×256
    dummy = torch.randn(2, 3, 256, 256)

    print("=== UNet++ (standard, no deep supervision) ===")
    model = UNetPP(in_channels=3, out_channels=1, depth=4, base_filters=32)
    out = model(dummy)
    print(f"  Input  : {dummy.shape}")
    print(f"  Output : {out.shape}")            # expects (2, 1, 256, 256)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    print("\n=== UNet++ (deep supervision) ===")
    model_ds = UNetPP(in_channels=3, out_channels=1, depth=4, base_filters=32, deep_supervision=True)
    out_ds = model_ds(dummy)
    print(f"  Input  : {dummy.shape}")
    print(f"  Output : {out_ds.shape}")         # expects (2, 1, 256, 256)

    total_params = sum(p.numel() for p in model_ds.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    print("="*60)
    print(model)
    print("="*60)
    print(model_ds)