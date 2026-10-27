import math
import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")),
)

from mola_pq import (
    MotionLatentBfp64Pq,
    Q30_ONE,
    QuantumSANConv1dPq,
    pose_clip_pq,
    sigmoid_pq_q16,
)


class TestSonyMolaPq(unittest.TestCase):
    def test_pose_clip_pq_variable_length_gate(self):
        # 20 frames of 264-dim HumanML3D (263 motion + 1 activation logit)
        # Frames 0..13 have logit +8.5 (sigmoid > 0.9998 > 0.999)
        # Frame 14 drops to logit +3.0 (sigmoid ~= 0.952 < 0.999)
        seq = []
        for t in range(20):
            logit = 8.5 if t < 14 else 3.0
            seq.append([float(t), 0.5, -0.2, logit])
        clipped = pose_clip_pq(seq)
        self.assertEqual(len(clipped), 14)
        self.assertEqual(len(clipped[0]), 3)

    def test_motion_latent_bfp64_compression(self):
        # 32 frames x 16 latent channels = 512 floats (2048 bytes FP32)
        latents = [
            [math.sin(0.11 * (t + 1) * (c + 1)) * 2.4 for c in range(16)]
            for t in range(32)
        ]
        packed8 = MotionLatentBfp64Pq.compress(latents, n_bits=8)
        self.assertGreater(packed8.fp32_byte_size / packed8.byte_size, 3.9)
        rec8 = packed8.decompress()
        for r_orig, r_rec in zip(latents, rec8):
            for a, b in zip(r_orig, r_rec):
                self.assertAlmostEqual(a, b, delta=0.04)

    def test_quantum_san_conv1d_pq_popcnt_norm_and_forward(self):
        in_ch = 32
        out_ch = 8
        k_size = 3
        weights = [
            [
                [math.sin(0.17 * (o + 1) * (i + 1) + 0.3 * k) for k in range(k_size)]
                for i in range(in_ch)
            ]
            for o in range(out_ch)
        ]
        san_conv = QuantumSANConv1dPq(weights, stride=1, padding=1)
        for c in range(out_ch):
            pop = san_conv.active_counts[c]
            self.assertGreater(pop, 0)
            unit_norm = (san_conv.inv_norm_q30[c] / float(Q30_ONE)) * math.sqrt(pop)
            self.assertAlmostEqual(unit_norm, 1.0, delta=5e-9)

        x = [[math.cos(0.09 * (ch + 1) * (t + 1)) for t in range(12)] for ch in range(in_ch)]
        y = san_conv.forward(x)
        self.assertEqual(len(y), out_ch)
        self.assertEqual(len(y[0]), 12)


if __name__ == "__main__":
    unittest.main()
