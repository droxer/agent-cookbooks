def check_image_exists(url: str, timeout: float = 5.0) -> bool:
    """
    检查图片是否存在（可返回200且能被Pillow识别）
    """
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return False
        # 检查是否可解析为图片
        try:
            Image.open(BytesIO(r.content)).verify()
        except Exception:
            # 如果是SVG等非位图，可简单检测content-type
            if "svg" not in r.headers.get("Content-Type", "").lower():
                return False
        return True
    except Exception:
        return False

# === 清洗数据 ===
valid_docs = []
for d in docs:
    if check_image_exists(d["image_url"]):
        print(f"✅ [OK] {d['id']} - {d['metadata']['topic']}")
        valid_docs.append(d)
    else:
        print(f"⚠️ [SKIP] {d['id']} - Invalid or missing image: {d['image_url']}")

print(f"\n共保留 {len(valid_docs)} 条有效数据。")