"""Unit and end-to-end tests for the watermark service.

Run with:  python -m unittest discover -s tests -v
"""

import base64
import hashlib
import io
import json
import os
import random
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.rs import rs_encode, rs_decode, RSDecodeError            # noqa: E402
from app.watermark import (embed, extract, pack_payload,           # noqa: E402
                           unpack_payload, PayloadError)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_test_image(seed=42, size=(700, 900)):
    rng = np.random.RandomState(seed)
    h, w = size
    y, x = np.mgrid[0:h, 0:w]
    img = 0.4 * x / w + 0.3 * y / h + 0.3 * np.sin(x / 37.0) * np.cos(y / 29.0)
    img = (img - img.min()) / (img.max() - img.min()) * 255
    arr = np.clip(img + rng.randn(h, w) * 12, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "L").convert("RGB")


def jpeg(img, quality=60):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(buf).convert("L")


def b64_of_image(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, fmt)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Reed-Solomon codec
# ---------------------------------------------------------------------------

class TestRS(unittest.TestCase):
    def test_roundtrip_and_correction(self):
        rng = random.Random(1)
        k, nsym = 81, 142
        for _ in range(50):
            msg = [rng.randrange(256) for _ in range(k)]
            code = rs_encode(msg, nsym)
            dec, nerr = rs_decode(code, nsym)
            self.assertEqual(dec, msg)
            self.assertEqual(nerr, 0)
            # corrupt up to nsym/2 symbols -> must still decode
            c2 = list(code)
            for pos in rng.sample(range(len(code)), nsym // 2):
                c2[pos] ^= rng.randrange(1, 256)
            dec, _ = rs_decode(c2, nsym)
            self.assertEqual(dec, msg)

    def test_beyond_capacity_raises(self):
        rng = random.Random(2)
        msg = [rng.randrange(256) for _ in range(81)]
        code = rs_encode(msg, 142)
        c2 = list(code)
        for pos in rng.sample(range(len(code)), 120):
            c2[pos] ^= 0x5A
        with self.assertRaises(RSDecodeError):
            rs_decode(c2, 142)


# ---------------------------------------------------------------------------
# payload packing
# ---------------------------------------------------------------------------

class TestPayload(unittest.TestCase):
    def test_pack_unpack(self):
        payload = pack_payload("hello-水印", 7, b"\x01" * 8)
        text, version, digest = unpack_payload(payload)
        self.assertEqual(text, "hello-水印")
        self.assertEqual(version, 7)
        self.assertEqual(digest, b"\x01" * 8)

    def test_max_length(self):
        payload = pack_payload("x" * 64, 1, b"\x00" * 8)
        text, _, _ = unpack_payload(payload)
        self.assertEqual(text, "x" * 64)

    def test_too_long(self):
        with self.assertRaises(PayloadError):
            pack_payload("x" * 65, 1, b"\x00" * 8)

    def test_bad_version(self):
        for bad in (0, -1, 2 ** 31):
            with self.assertRaises(PayloadError):
                pack_payload("w", bad, b"\x00" * 8)


# ---------------------------------------------------------------------------
# embed / extract robustness
# ---------------------------------------------------------------------------

class TestWatermark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img = make_test_image()
        cls.raw = cls.img.tobytes()
        cls.digest = hashlib.sha256(cls.raw).digest()[:8]
        cls.wm, cls.alpha = embed("robustness-测试", 99, cls.digest, cls.img)

    def _check(self, attacked, tag):
        result = extract(attacked)
        text, version, digest = unpack_payload(result["payload"])
        self.assertEqual(text, "robustness-测试", tag)
        self.assertEqual(version, 99, tag)
        self.assertEqual(digest, self.digest, tag)
        self.assertGreater(result["confidence"], 0.8, tag)

    def test_clean(self):
        self._check(self.wm, "clean")

    def test_jpeg60(self):
        self._check(jpeg(self.wm, 60), "jpeg60")

    def test_scale_down(self):
        self._check(self.wm.resize((768, 768), Image.BICUBIC), "scale 0.75")

    def test_scale_up(self):
        self._check(self.wm.resize((1536, 1536), Image.BICUBIC), "scale 1.5")

    def test_center_crop_15(self):
        self._check(self.wm.crop((77, 77, 947, 947)), "crop 15%")

    def test_scale_and_jpeg(self):
        attacked = jpeg(self.wm.resize((768, 768), Image.BICUBIC), 60)
        self._check(attacked, "scale 0.75 + jpeg60")

    def test_crop_and_jpeg(self):
        attacked = jpeg(self.wm.crop((77, 77, 947, 947)), 60)
        self._check(attacked, "crop 15% + jpeg60")

    def test_offsets_reported(self):
        result = extract(self.wm.crop((77, 77, 947, 947)))
        self.assertAlmostEqual(result["offset_x"], -77, delta=3)
        self.assertAlmostEqual(result["offset_y"], -77, delta=3)


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        os.environ["DB_PATH"] = os.path.join(cls.tmp.name, "test.db")
        # re-import to pick up the temp DB path
        for mod in ("app.main", "app.db"):
            sys.modules.pop(mod, None)
        from app.main import app, db
        cls.app = app.test_client()
        cls.db = db

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _create_asset(self, watermark="api-测试", version=5, seed=11):
        img = make_test_image(seed=seed)
        png = io.BytesIO()
        img.save(png, "PNG")
        raw_png = png.getvalue()
        resp = self.app.post("/assets", json={
            "image_base64": base64.b64encode(raw_png).decode(),
            "watermark": watermark,
            "version": version,
        })
        self.assertEqual(resp.status_code, 201, resp.get_json())
        return resp.get_json(), hashlib.sha256(raw_png).hexdigest()

    def test_create_and_extract_clean(self):
        created, sha = self._create_asset()
        self.assertEqual(created["sha256"], sha)
        self.assertIn("asset_id", created)
        self.assertIn("watermarked_base64", created)

        resp = self.app.post(f"/assets/{created['asset_id']}/extract", json={
            "image_base64": created["watermarked_base64"],
        })
        self.assertEqual(resp.status_code, 200, resp.get_json())
        data = resp.get_json()
        self.assertEqual(data["watermark"], "api-测试")
        self.assertTrue(data["integrity"])
        self.assertGreater(data["confidence"], 0.9)
        self.assertAlmostEqual(data["offset_x"], 0, delta=3)
        self.assertAlmostEqual(data["offset_y"], 0, delta=3)

    def test_extract_after_jpeg_and_scale(self):
        created, _ = self._create_asset()
        wm_bytes = base64.b64decode(created["watermarked_base64"])
        wm_img = Image.open(io.BytesIO(wm_bytes)).convert("L")
        attacked = jpeg(wm_img.resize((768, 768), Image.BICUBIC), 60)
        resp = self.app.post(f"/assets/{created['asset_id']}/extract", json={
            "image_base64": b64_of_image(attacked),
        })
        self.assertEqual(resp.status_code, 200, resp.get_json())
        self.assertEqual(resp.get_json()["watermark"], "api-测试")

    def test_version_mismatch(self):
        created, _ = self._create_asset(version=5)
        # tamper with the stored version
        import sqlite3
        with sqlite3.connect(self.db.path) as conn:
            conn.execute("UPDATE assets SET version = 6 WHERE asset_id = ?",
                         (created["asset_id"],))
        resp = self.app.post(f"/assets/{created['asset_id']}/extract", json={
            "image_base64": created["watermarked_base64"],
        })
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.get_json()["error"], "version_mismatch")

    def test_digest_mismatch(self):
        created_a, _ = self._create_asset(version=5, seed=11)
        created_b, _ = self._create_asset(version=5, seed=22)
        # extract B's image against A's record -> digest must not match
        resp = self.app.post(f"/assets/{created_a['asset_id']}/extract", json={
            "image_base64": created_b["watermarked_base64"],
        })
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.get_json()["error"], "digest_mismatch")

    def test_asset_not_found(self):
        resp = self.app.post("/assets/deadbeef/extract", json={
            "image_base64": b64_of_image(make_test_image()),
        })
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.get_json()["error"], "asset_not_found")

    def test_unwatermarked_image(self):
        created, _ = self._create_asset()
        plain = make_test_image(seed=77).convert("L").resize((1024, 1024))
        resp = self.app.post(f"/assets/{created['asset_id']}/extract", json={
            "image_base64": b64_of_image(plain),
        })
        self.assertEqual(resp.status_code, 422)
        self.assertIn(resp.get_json()["error"],
                      ("watermark_not_detected", "watermark_undecodable"))

    def test_validation_errors(self):
        img64 = b64_of_image(make_test_image())
        # watermark too long
        resp = self.app.post("/assets", json={
            "image_base64": img64, "watermark": "x" * 65, "version": 1})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["error"], "watermark_too_long")
        # utf-8 length counted in bytes, not characters
        resp = self.app.post("/assets", json={
            "image_base64": img64, "watermark": "水" * 22, "version": 1})
        self.assertEqual(resp.status_code, 400)
        # bad versions
        for bad in (0, -3, 1.5, "2", True):
            resp = self.app.post("/assets", json={
                "image_base64": img64, "watermark": "w", "version": bad})
            self.assertEqual(resp.status_code, 400, f"version={bad!r}")
        # bad base64
        resp = self.app.post("/assets", json={
            "image_base64": "!!!not-base64!!!", "watermark": "w",
            "version": 1})
        self.assertEqual(resp.status_code, 400)
        # not an image
        resp = self.app.post("/assets", json={
            "image_base64": base64.b64encode(b"hello").decode(),
            "watermark": "w", "version": 1})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.get_json()["error"], "invalid_image")

    def test_health(self):
        resp = self.app.get("/health")
        self.assertEqual(resp.status_code, 200)


if __name__ == "__main__":
    unittest.main()
