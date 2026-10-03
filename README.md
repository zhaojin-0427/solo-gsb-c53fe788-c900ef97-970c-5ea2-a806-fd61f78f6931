# 图像水印验证 API

基于 Python + Pillow + SQLite 的图像水印注册与验证服务。将水印文本、版本号和原图摘要通过**分块频域编码**（8×8 块 DCT 中频系数 + 导频同步 + Reed-Solomon 纠错）嵌入图像，之后即使图像经过 JPEG 压缩（质量 60）、0.75～1.5 倍缩放、不超过 15% 的中心裁剪（含这些攻击的组合），仍能恢复水印载荷并校验完整性。

## 快速启动（Docker Compose）

```bash
docker compose up --build
```

服务监听 `http://localhost:8000`，SQLite 数据库持久化在 `watermark-data` 卷中。

健康检查：

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### 本地开发（不用 Docker）

```bash
pip install -r requirements.txt
python -m app.main          # 监听 0.0.0.0:8000，DB 在 ./data/watermark.db
```

环境变量：`PORT`（默认 8000）、`DB_PATH`（默认 `./data/watermark.db`，容器内为 `/data/watermark.db`）。

## API 使用

### 1. 注册资产并嵌入水印 — `POST /assets`

请求体（JSON）：

| 字段 | 类型 | 说明 |
|---|---|---|
| `image_base64` | string | 原始图像的 Base64（PNG/JPEG 等，可带 data-URL 前缀） |
| `watermark` | string | 水印文本，UTF-8 编码后 **≤ 64 字节** |
| `version` | int | 正整数（< 2^31） |

```bash
IMG_B64=$(base64 -w0 photo.png)
curl -X POST http://localhost:8000/assets \
  -H 'Content-Type: application/json' \
  -d "{\"image_base64\": \"$IMG_B64\", \"watermark\": \"机密-批次A\", \"version\": 3}"
```

响应 `201 Created`：

```json
{
  "asset_id": "9f2c1c...",
  "sha256": "b5f1...（原图字节的 SHA-256，hex）",
  "version": 3,
  "watermark": "机密-批次A",
  "canvas_size": 1024,
  "watermarked_base64": "iVBOR...（嵌入水印后的 PNG，Base64）"
}
```

服务端处理：图像转灰度并缩放到 1024×1024 固定画布 → 计算原图 SHA-256 → 将
`水印 + 版本 + 摘要(8B) + CRC32` 组成 81 字节载荷，经 RS(223,81) 纠错编码和 32bit
同步字后，以伪随机置换分散嵌入 8×8 块 DCT 的 (3,2)/(2,3) 中频系数（每比特重复 18
个槽位，幅度随内容自适应 32～64），并叠加导频音用于几何同步。原图（灰度画布）、
编码参数和版本存入 SQLite。

### 2. 提取并验证水印 — `POST /assets/{asset_id}/extract`

对待验证图像（可以是经过 JPEG/缩放/裁剪的版本）提取水印并与注册信息比对。

```bash
ATTACKED_B64=$(base64 -w0 photo_transformed.jpg)
curl -X POST http://localhost:8000/assets/9f2c1c.../extract \
  -H 'Content-Type: application/json' \
  -d "{\"image_base64\": \"$ATTACKED_B64\"}"
```

响应 `200 OK`：

```json
{
  "watermark": "机密-批次A",
  "confidence": 0.998,
  "offset_x": -58.0,
  "offset_y": -58.0,
  "integrity": true,
  "version": 3,
  "scale": 0.7504,
  "sync_score": 0.92,
  "rs_corrected_bytes": 21
}
```

| 字段 | 说明 |
|---|---|
| `watermark` | 恢复出的水印文本 |
| `confidence` | 接收比特与译码后码字的一致比例（0~1） |
| `offset_x` / `offset_y` | 检测到的画布原点在被测图像中的偏移（像素）；中心裁剪 15% 时约为 -77 |
| `integrity` | 载荷 CRC、版本、摘要全部校验通过时为 `true` |
| `scale` | 检测到的几何缩放比例（附加信息） |

### 3. 查询资产元数据 — `GET /assets/{asset_id}`

返回 `asset_id / sha256 / version / watermark / canvas_size / alpha / params / created_at`。

## 错误响应

所有错误均为 JSON：`{"error": "<code>", "message": "<说明>"}`。

| HTTP | code | 含义 |
|---|---|---|
| 400 | `invalid_request` / `invalid_base64` / `invalid_image` | 请求体、Base64 或图像数据非法 |
| 400 | `watermark_too_long` | 水印超过 64 字节 UTF-8 |
| 400 | `invalid_version` | version 不是正整数 |
| 404 | `asset_not_found` | asset_id 不存在 |
| 409 | `version_mismatch` | 解码出的版本与登记版本不一致 |
| 409 | `digest_mismatch` | 解码出的摘要与原图 SHA-256 不一致（图像被替换/篡改） |
| 422 | `watermark_not_detected` | 找不到水印同步信号（图像未嵌入水印或破坏过重） |
| 422 | `watermark_undecodable` | 信号存在但载荷损坏超出纠错能力 |

## 鲁棒性

在以下攻击及组合下可恢复载荷（测试见 `tests/test_watermark.py`）：

- JPEG 质量 60（实测质量 50 仍可通过）
- 0.75×～1.5× 缩放（含非整数倍，如 0.83、1.27）
- 中心裁剪 ≤ 15%
- 组合攻击：缩放 + JPEG、裁剪 + JPEG、0.75× + 裁剪 15% + JPEG 60

提取耗时约 0.3～1 秒/次（视图像尺寸而定）。

## 运行测试

```bash
python -m unittest discover -s tests -v
```

覆盖：RS 纠错编解码、载荷打包、嵌入/提取在各类攻击下的恢复、API 端到端
（注册、提取、版本/摘要不匹配、非法输入）。

## 项目结构

```
app/
  main.py        Flask 路由与请求校验
  watermark.py   水印嵌入/提取（DCT、导频同步、几何搜索）
  rs.py          Reed-Solomon over GF(2^8)（BM + Chien + Forney）
  db.py          SQLite 资产存储
tests/
  test_watermark.py
Dockerfile
docker-compose.yml
requirements.txt
```
