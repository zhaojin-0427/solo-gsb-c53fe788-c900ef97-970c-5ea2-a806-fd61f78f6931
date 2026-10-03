"""HTTP API for the image-watermark verification service.

Endpoints
---------
POST /assets                  embed a watermark, register the asset
POST /assets/<id>/extract     verify/extract a watermark from a
                              (possibly transformed) image
GET  /assets/<id>             asset metadata
GET  /health                  liveness probe
"""

import base64
import binascii
import hashlib
import io
import json
import os
import uuid

from flask import Flask, jsonify, request
from PIL import Image

from .db import Database
from .rs import RSDecodeError
from .watermark import (CANVAS_SIZE, MAX_WATERMARK_BYTES, NotDetectedError,
                        PayloadError, embed, extract, unpack_payload)

DB_PATH = os.environ.get("DB_PATH", os.path.join("data", "watermark.db"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 MB of JSON
db = Database(DB_PATH)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class ApiError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@app.errorhandler(ApiError)
def _handle_api_error(err):
    return jsonify({"error": err.code, "message": err.message}), err.status


@app.errorhandler(404)
def _handle_404(_):
    return jsonify({"error": "not_found", "message": "resource not found"}), 404


@app.errorhandler(413)
def _handle_413(_):
    return jsonify({"error": "payload_too_large",
                    "message": "request body too large"}), 413


@app.errorhandler(500)
def _handle_500(err):
    return jsonify({"error": "internal_error",
                    "message": "unexpected server error"}), 500


def _json_body():
    if not request.is_json:
        raise ApiError(400, "invalid_request",
                       "Content-Type must be application/json")
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ApiError(400, "invalid_request", "request body must be a JSON object")
    return body


def _decode_image_field(body, field="image_base64"):
    data = body.get(field)
    if not isinstance(data, str) or not data:
        raise ApiError(400, "invalid_request", f"'{field}' must be a base64 string")
    if data.startswith("data:"):  # tolerate data-URL prefixes
        comma = data.find(",")
        data = data[comma + 1:] if comma >= 0 else data
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        raise ApiError(400, "invalid_base64",
                       f"'{field}' is not valid base64")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        raise ApiError(400, "invalid_image",
                       "decoded bytes are not a readable image")
    return img, raw


def _validate_watermark(value):
    if not isinstance(value, str):
        raise ApiError(400, "invalid_watermark", "'watermark' must be a string")
    size = len(value.encode("utf-8"))
    if size > MAX_WATERMARK_BYTES:
        raise ApiError(400, "watermark_too_long",
                       f"watermark is {size} bytes UTF-8, "
                       f"max {MAX_WATERMARK_BYTES}")
    return value


def _validate_version(value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ApiError(400, "invalid_version",
                       "'version' must be a positive integer")
    if not (1 <= value <= 0x7FFFFFFF):
        raise ApiError(400, "invalid_version",
                       "'version' must be a positive integer (< 2^31)")
    return value


def _png_bytes(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/assets")
def create_asset():
    body = _json_body()
    img, raw = _decode_image_field(body)
    watermark = _validate_watermark(body.get("watermark"))
    version = _validate_version(body.get("version"))

    sha256 = hashlib.sha256(raw).hexdigest()
    digest8 = hashlib.sha256(raw).digest()[:8]

    gray = img.convert("L")
    watermarked, alpha = embed(watermark, version, digest8, gray)

    asset_id = uuid.uuid4().hex
    params = {
        "canvas_size": CANVAS_SIZE,
        "alpha": alpha,
        "scheme": "block-dct(3,2)+(2,3), rs(223,81), rep=18, pilot(2/8,1/8)",
    }
    db.insert_asset(
        asset_id=asset_id,
        sha256=sha256,
        version=version,
        watermark=watermark,
        canvas_size=CANVAS_SIZE,
        alpha=alpha,
        params_json=json.dumps(params),
        original_png=raw,  # the original uploaded image, as received
        watermarked_png=_png_bytes(watermarked),
    )

    return jsonify({
        "asset_id": asset_id,
        "sha256": sha256,
        "version": version,
        "watermark": watermark,
        "canvas_size": CANVAS_SIZE,
        "watermarked_base64": base64.b64encode(_png_bytes(watermarked)).decode(),
    }), 201


@app.get("/assets/<asset_id>")
def asset_info(asset_id):
    asset = db.get_asset(asset_id)
    if asset is None:
        return jsonify({"error": "asset_not_found",
                        "message": f"no asset with id '{asset_id}'"}), 404
    asset["params"] = json.loads(asset["params"])
    return jsonify(asset)


@app.post("/assets/<asset_id>/extract")
def extract_asset(asset_id):
    asset = db.get_asset(asset_id)
    if asset is None:
        return jsonify({"error": "asset_not_found",
                        "message": f"no asset with id '{asset_id}'"}), 404

    body = _json_body()
    img, _raw = _decode_image_field(body)

    try:
        result = extract(img)
    except NotDetectedError as exc:
        raise ApiError(422, "watermark_not_detected",
                       f"no watermark synchronisation found: {exc}")
    except RSDecodeError:
        raise ApiError(422, "watermark_undecodable",
                       "watermark payload damaged beyond error correction")
    except PayloadError as exc:
        raise ApiError(422, "watermark_undecodable", str(exc))

    wm_text, version, digest8 = unpack_payload(result["payload"])

    if version != asset["version"]:
        raise ApiError(409, "version_mismatch",
                       f"payload version {version} does not match stored "
                       f"version {asset['version']}")
    if digest8.hex() != asset["sha256"][:16]:
        raise ApiError(409, "digest_mismatch",
                       "payload digest does not match the stored SHA-256 of "
                       "the original image")

    return jsonify({
        "watermark": wm_text,
        "confidence": round(result["confidence"], 4),
        "offset_x": round(result["offset_x"], 2),
        "offset_y": round(result["offset_y"], 2),
        "integrity": True,
        "version": version,
        "scale": round(result["scale"], 4),
        "sync_score": round(result["sync_score"], 4),
        "rs_corrected_bytes": result["rs_corrected"],
    })


def main():
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()
