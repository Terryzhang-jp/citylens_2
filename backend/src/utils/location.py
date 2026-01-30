"""
位置服务 - 使用 OpenStreetMap Overpass API 获取位置信息
"""

import httpx
from typing import Optional


# Overpass API 备用端点
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


def get_nearby_pois(lat: float, lon: float, radius: int = 50) -> list[dict]:
    """
    使用 Overpass API 查询坐标附近的 POI（兴趣点）

    Args:
        lat: 纬度
        lon: 经度
        radius: 搜索半径（米）

    Returns:
        POI 列表，每个包含 name, type, distance 等信息
    """
    # Overpass QL 查询：获取附近有名字的建筑、商店、地标等
    query = f"""
    [out:json][timeout:10];
    (
      // 有名字的建筑
      way["building"]["name"](around:{radius},{lat},{lon});
      relation["building"]["name"](around:{radius},{lat},{lon});
      // 商店、餐厅等
      node["shop"]["name"](around:{radius},{lat},{lon});
      node["amenity"]["name"](around:{radius},{lat},{lon});
      // 旅游景点、地标
      node["tourism"]["name"](around:{radius},{lat},{lon});
      way["tourism"]["name"](around:{radius},{lat},{lon});
      // 历史建筑
      node["historic"]["name"](around:{radius},{lat},{lon});
      way["historic"]["name"](around:{radius},{lat},{lon});
      // 办公楼、商业建筑
      way["office"]["name"](around:{radius},{lat},{lon});
      way["landuse"="commercial"]["name"](around:{radius},{lat},{lon});
    );
    out center tags;
    """

    print(f"[Location] 查询 ({lat}, {lon}) 半径 {radius}m 内的 POI...")

    # 尝试多个端点
    data = None
    for endpoint in OVERPASS_ENDPOINTS:
        try:
            resp = httpx.post(
                endpoint,
                data={"data": query},
                timeout=20.0,
            )
            resp.raise_for_status()
            data = resp.json()
            print(f"[Location] 使用端点: {endpoint.split('/')[2]}")
            break  # 成功就退出
        except Exception as e:
            print(f"[Location] {endpoint.split('/')[2]} 失败: {type(e).__name__}")
            continue

    if not data:
        print("[Location] 所有 Overpass 端点都失败")
        return []

    pois = []
    seen_names = set()  # 去重

    for element in data.get("elements", []):
        tags = element.get("tags", {})
        name = tags.get("name") or tags.get("name:en") or tags.get("name:ja")

        if not name or name in seen_names:
            continue

        seen_names.add(name)

        # 确定类型
        poi_type = (
            tags.get("building") or
            tags.get("shop") or
            tags.get("amenity") or
            tags.get("tourism") or
            tags.get("historic") or
            tags.get("office") or
            "unknown"
        )

        # 获取坐标（way/relation 用 center）
        if element.get("type") == "node":
            poi_lat = element.get("lat")
            poi_lon = element.get("lon")
        else:
            center = element.get("center", {})
            poi_lat = center.get("lat")
            poi_lon = center.get("lon")

        pois.append({
            "name": name,
            "name_en": tags.get("name:en", ""),
            "name_ja": tags.get("name:ja", ""),
            "type": poi_type,
            "architect": tags.get("architect", ""),
            "opening_date": tags.get("opening_date") or tags.get("start_date", ""),
            "description": tags.get("description", ""),
            "wikipedia": tags.get("wikipedia", ""),
            "lat": poi_lat,
            "lon": poi_lon,
        })

    print(f"[Location] 找到 {len(pois)} 个 POI")
    return pois


def format_pois_for_prompt(pois: list[dict]) -> str:
    """
    将 POI 列表格式化为 prompt 可用的文本
    """
    if not pois:
        return "未找到附近的已知地点信息。"

    lines = ["该位置附近的已知地点："]
    for poi in pois[:10]:  # 最多 10 个
        line = f"- {poi['name']}"
        if poi.get("name_en"):
            line += f" ({poi['name_en']})"
        if poi.get("type") and poi["type"] != "yes":
            line += f" [{poi['type']}]"
        if poi.get("architect"):
            line += f" 建筑师: {poi['architect']}"
        if poi.get("opening_date"):
            line += f" 开业: {poi['opening_date']}"
        lines.append(line)

    return "\n".join(lines)


# 测试用的表参道坐标
OMOTESANDO_COORDS = (35.6654, 139.7121)  # 表参道十字路口附近


if __name__ == "__main__":
    # 测试查询
    lat, lon = OMOTESANDO_COORDS
    pois = get_nearby_pois(lat, lon, radius=100)

    print("\n" + "=" * 50)
    print(format_pois_for_prompt(pois))
    print("=" * 50)

    # 打印详细信息
    for poi in pois:
        print(f"\n{poi['name']}")
        for k, v in poi.items():
            if v and k != "name":
                print(f"  {k}: {v}")
