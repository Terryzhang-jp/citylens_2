"""
EXIF 数据提取工具
从图片中提取 GPS 坐标、拍摄时间等元数据
"""

from io import BytesIO
from typing import Optional, TypedDict
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS


class GPSData(TypedDict):
    lat: float
    lon: float


class ExifData(TypedDict, total=False):
    gps: Optional[GPSData]
    datetime: Optional[str]
    camera: Optional[str]
    orientation: Optional[int]


def _convert_to_degrees(value) -> float:
    """将 EXIF GPS 坐标转换为十进制度数"""
    try:
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)
    except (TypeError, IndexError, ZeroDivisionError):
        return 0.0


def _extract_gps(exif_data: dict) -> Optional[GPSData]:
    """从 EXIF 数据中提取 GPS 坐标"""
    gps_info = exif_data.get("GPSInfo")
    if not gps_info:
        return None

    # 解析 GPS tags
    gps_data = {}
    for tag_id, value in gps_info.items():
        tag_name = GPSTAGS.get(tag_id, tag_id)
        gps_data[tag_name] = value

    # 提取纬度
    lat = gps_data.get("GPSLatitude")
    lat_ref = gps_data.get("GPSLatitudeRef", "N")

    # 提取经度
    lon = gps_data.get("GPSLongitude")
    lon_ref = gps_data.get("GPSLongitudeRef", "E")

    if not lat or not lon:
        return None

    try:
        lat_decimal = _convert_to_degrees(lat)
        lon_decimal = _convert_to_degrees(lon)

        # 处理南纬和西经
        if lat_ref == "S":
            lat_decimal = -lat_decimal
        if lon_ref == "W":
            lon_decimal = -lon_decimal

        # 验证坐标范围
        if not (-90 <= lat_decimal <= 90) or not (-180 <= lon_decimal <= 180):
            return None

        return GPSData(lat=round(lat_decimal, 6), lon=round(lon_decimal, 6))
    except (TypeError, ValueError):
        return None


def extract_exif(image_data: bytes) -> ExifData:
    """
    从图片数据中提取 EXIF 信息

    Args:
        image_data: 图片的字节数据

    Returns:
        ExifData 字典，包含:
        - gps: {lat, lon} 或 None
        - datetime: 拍摄时间字符串 或 None
        - camera: 相机型号 或 None
        - orientation: 图片方向 或 None

    Note:
        该函数不会抛出异常，解析失败时返回空字典
    """
    result: ExifData = {}

    try:
        image = Image.open(BytesIO(image_data))
        exif_raw = image._getexif()

        if not exif_raw:
            return result

        # 转换 EXIF tag IDs 为可读名称
        exif_data = {}
        for tag_id, value in exif_raw.items():
            tag_name = TAGS.get(tag_id, tag_id)
            exif_data[tag_name] = value

        # 提取 GPS
        gps = _extract_gps(exif_data)
        if gps:
            result["gps"] = gps

        # 提取拍摄时间
        datetime_str = (
            exif_data.get("DateTimeOriginal") or
            exif_data.get("DateTime") or
            exif_data.get("DateTimeDigitized")
        )
        if datetime_str and isinstance(datetime_str, str):
            # 格式化: "2024:01:15 14:30:00" → "2024-01-15 14:30"
            try:
                formatted = datetime_str.replace(":", "-", 2)[:16]
                result["datetime"] = formatted
            except (IndexError, AttributeError):
                result["datetime"] = datetime_str

        # 提取相机信息
        make = exif_data.get("Make", "")
        model = exif_data.get("Model", "")
        if make or model:
            # 避免重复 (如 "Apple iPhone 15" vs "Apple Apple iPhone 15")
            if model and make and model.startswith(make):
                result["camera"] = model.strip()
            elif make and model:
                result["camera"] = f"{make} {model}".strip()
            else:
                result["camera"] = (make or model).strip()

        # 提取方向
        orientation = exif_data.get("Orientation")
        if orientation and isinstance(orientation, int):
            result["orientation"] = orientation

    except Exception as e:
        # 静默处理所有错误，返回已提取的数据
        print(f"[EXIF] 提取警告: {type(e).__name__}: {e}")

    return result


def format_exif_for_display(exif: ExifData) -> dict:
    """
    格式化 EXIF 数据用于前端显示

    Returns:
        {
            "location": "35.6654, 139.7121" 或 None,
            "datetime": "2024-01-15 14:30" 或 None,
            "camera": "iPhone 15 Pro" 或 None
        }
    """
    result = {}

    if exif.get("gps"):
        gps = exif["gps"]
        result["location"] = f"{gps['lat']}, {gps['lon']}"
        result["lat"] = gps["lat"]
        result["lon"] = gps["lon"]

    if exif.get("datetime"):
        result["datetime"] = exif["datetime"]

    if exif.get("camera"):
        result["camera"] = exif["camera"]

    return result


if __name__ == "__main__":
    # 测试代码
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            data = f.read()

        exif = extract_exif(data)
        print("Raw EXIF:", exif)
        print("Formatted:", format_exif_for_display(exif))
    else:
        print("Usage: python exif.py <image_file>")
