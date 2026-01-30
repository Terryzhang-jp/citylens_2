"""
MobileSAM 分割工具
使用 Gemini 返回的 bounding box 进行精细分割
"""

import io
import base64
from typing import Optional, TypedDict
from PIL import Image
import numpy as np

# 延迟导入，避免启动时加载模型
_sam_model = None


class BoundingBox(TypedDict):
    x1: int  # 0-1000
    y1: int  # 0-1000
    x2: int  # 0-1000
    y2: int  # 0-1000


def _get_sam_model():
    """
    延迟加载 MobileSAM 模型

    Ultralytics SAM 会自动下载模型文件（如果不存在）:
    - 模型来源: https://github.com/ultralytics/assets/releases
    - 文件大小: ~24MB
    - 下载位置: 当前工作目录或 ~/.ultralytics/
    """
    global _sam_model
    if _sam_model is None:
        try:
            from ultralytics import SAM
            import os

            model_path = "mobile_sam.pt"

            # 检查模型是否存在
            if not os.path.exists(model_path):
                print("[Segmentation] MobileSAM 模型不存在，将自动下载...")
                print("[Segmentation] 下载源: Ultralytics GitHub Releases (~24MB)")
            else:
                print("[Segmentation] 加载本地 MobileSAM 模型...")

            # Ultralytics SAM 会自动下载如果文件不存在
            _sam_model = SAM(model_path)
            print("[Segmentation] MobileSAM 模型加载完成 ✓")

        except Exception as e:
            print(f"[Segmentation] 加载 MobileSAM 失败: {e}")
            print("[Segmentation] 将使用简单裁剪作为回退方案")
            return None
    return _sam_model


def _bbox_to_pixels(bbox: BoundingBox, width: int, height: int) -> list[int]:
    """
    将 0-1000 归一化坐标转换为像素坐标

    Args:
        bbox: {x1, y1, x2, y2} 范围 0-1000
        width: 图片宽度
        height: 图片高度

    Returns:
        [x1, y1, x2, y2] 像素坐标
    """
    return [
        int(bbox["x1"] * width / 1000),
        int(bbox["y1"] * height / 1000),
        int(bbox["x2"] * width / 1000),
        int(bbox["y2"] * height / 1000),
    ]


def segment_region(
    image_data: bytes,
    bbox: BoundingBox,
    output_format: str = "png",
    add_padding: int = 10,
) -> Optional[str]:
    """
    使用 MobileSAM 从 bounding box 精细分割区域

    Args:
        image_data: 原图字节
        bbox: {"x1": 0-1000, "y1": 0-1000, "x2": 0-1000, "y2": 0-1000}
        output_format: 输出格式 "png" 或 "jpeg"
        add_padding: 在 bbox 周围添加的像素边距

    Returns:
        抠出区域的 base64 编码字符串，失败返回 None
    """
    try:
        # 加载图片
        image = Image.open(io.BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")

        width, height = image.size

        # 转换 bbox 坐标
        pixel_bbox = _bbox_to_pixels(bbox, width, height)

        # 获取 SAM 模型
        model = _get_sam_model()

        if model is None:
            # SAM 加载失败，回退到简单裁剪
            return _fallback_crop(image, pixel_bbox, add_padding, output_format)

        # 运行 SAM 分割
        image_np = np.array(image)
        results = model(image_np, bboxes=[pixel_bbox])

        if not results or len(results) == 0:
            return _fallback_crop(image, pixel_bbox, add_padding, output_format)

        # 获取第一个结果的 mask
        result = results[0]
        if result.masks is None or len(result.masks.data) == 0:
            return _fallback_crop(image, pixel_bbox, add_padding, output_format)

        mask = result.masks.data[0].cpu().numpy()

        # 应用 mask 创建透明背景图片
        cropped = _apply_mask_and_crop(image_np, mask, pixel_bbox, add_padding)

        # 编码为 base64
        return _encode_image(cropped, output_format)

    except Exception as e:
        print(f"[Segmentation] 分割失败: {e}")
        # 回退到简单裁剪
        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != "RGB":
                image = image.convert("RGB")
            pixel_bbox = _bbox_to_pixels(bbox, image.width, image.height)
            return _fallback_crop(image, pixel_bbox, add_padding, output_format)
        except Exception as e2:
            print(f"[Segmentation] 回退裁剪也失败: {e2}")
            return None


def _apply_mask_and_crop(
    image_np: np.ndarray,
    mask: np.ndarray,
    bbox: list[int],
    padding: int = 10,
) -> Image.Image:
    """
    应用 mask 并裁剪到 bbox 区域

    Args:
        image_np: RGB 图片数组
        mask: 二值 mask 数组
        bbox: [x1, y1, x2, y2] 像素坐标
        padding: 边距

    Returns:
        带透明背景的 PIL Image
    """
    h, w = image_np.shape[:2]

    # 添加 padding 到 bbox
    x1 = max(0, bbox[0] - padding)
    y1 = max(0, bbox[1] - padding)
    x2 = min(w, bbox[2] + padding)
    y2 = min(h, bbox[3] + padding)

    # 裁剪图片和 mask
    cropped_image = image_np[y1:y2, x1:x2]

    # 调整 mask 大小以匹配原图
    if mask.shape != (h, w):
        from PIL import Image as PILImage
        mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((w, h), PILImage.NEAREST)
        mask = np.array(mask_pil) / 255.0

    cropped_mask = mask[y1:y2, x1:x2]

    # 创建 RGBA 图片
    rgba = np.zeros((y2 - y1, x2 - x1, 4), dtype=np.uint8)
    rgba[:, :, :3] = cropped_image
    rgba[:, :, 3] = (cropped_mask * 255).astype(np.uint8)

    return Image.fromarray(rgba, "RGBA")


def _fallback_crop(
    image: Image.Image,
    bbox: list[int],
    padding: int,
    output_format: str,
) -> str:
    """
    SAM 不可用时的简单裁剪回退
    """
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.width, x2 + padding)
    y2 = min(image.height, y2 + padding)

    cropped = image.crop((x1, y1, x2, y2))
    return _encode_image(cropped, output_format)


def _encode_image(image: Image.Image, format: str = "png") -> str:
    """
    将 PIL Image 编码为 base64
    """
    buffer = io.BytesIO()

    if format.lower() == "png":
        # PNG 支持透明背景
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        image.save(buffer, format="PNG", optimize=True)
        mime = "image/png"
    else:
        # JPEG 不支持透明，转为 RGB
        if image.mode == "RGBA":
            # 创建白色背景
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buffer, format="JPEG", quality=90)
        mime = "image/jpeg"

    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def batch_segment_regions(
    image_data: bytes,
    bboxes: list[BoundingBox],
    output_format: str = "png",
) -> list[Optional[str]]:
    """
    批量分割多个区域

    Args:
        image_data: 原图字节
        bboxes: bbox 列表
        output_format: 输出格式

    Returns:
        base64 编码的图片列表，失败的为 None
    """
    results = []
    for bbox in bboxes:
        result = segment_region(image_data, bbox, output_format)
        results.append(result)
    return results


# 预热模型（可选，在启动时调用）
def warmup():
    """预热 MobileSAM 模型"""
    print("[Segmentation] 预热模型...")
    _get_sam_model()


if __name__ == "__main__":
    # 测试代码
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            image_data = f.read()

        # 测试分割中心区域
        test_bbox = BoundingBox(x1=250, y1=250, x2=750, y2=750)
        result = segment_region(image_data, test_bbox)

        if result:
            print(f"分割成功，base64 长度: {len(result)}")
            # 保存测试结果
            import re
            match = re.match(r"data:image/(\w+);base64,(.+)", result)
            if match:
                ext, b64_data = match.groups()
                with open(f"test_segment.{ext}", "wb") as f:
                    f.write(base64.b64decode(b64_data))
                print(f"已保存到 test_segment.{ext}")
        else:
            print("分割失败")
    else:
        print("Usage: python segmentation.py <image_file>")
