"""
Director Agent - 导演/调度者
分析图片内容，动态选择最合适的观察视角
"""

# Note: Only PERSPECTIVE_POOL is used in graph_v7.py


# 可用的专业视角池
PERSPECTIVE_POOL = {
    # 设计与艺术
    "designer": {
        "name": "设计师",
        "expertise": "工业设计、产品设计、平面设计、用户体验",
        "focus": "设计理念、材质选择、人机交互、功能与美学的平衡",
    },
    "architect": {
        "name": "建筑师",
        "expertise": "建筑设计、空间规划、结构美学",
        "focus": "建筑风格、空间关系、材质、光影、设计师意图",
    },
    "artist": {
        "name": "视觉艺术家",
        "expertise": "绘画、摄影、色彩理论、构图",
        "focus": "色彩搭配、光影层次、视觉节奏、美学价值",
    },

    # 科学与自然
    "botanist": {
        "name": "植物学家",
        "expertise": "植物分类、生态学、园艺",
        "focus": "植物种类、生长习性、生态意义、季节变化",
    },
    "physicist": {
        "name": "物理学家",
        "expertise": "光学、力学、材料科学",
        "focus": "光的折射反射、结构力学、材料特性、自然现象原理",
    },
    "meteorologist": {
        "name": "气象学家",
        "expertise": "天气、云层、大气现象",
        "focus": "云的类型、天气预兆、大气光学现象、气候特征",
    },
    "geologist": {
        "name": "地质学家",
        "expertise": "岩石、地形、地质历史",
        "focus": "岩石类型、地质构造、侵蚀痕迹、地质年代",
    },
    "biologist": {
        "name": "生物学家",
        "expertise": "动物行为、生态系统、进化",
        "focus": "动物行为、生态关系、适应性特征、生物多样性",
    },

    # 人文与社会
    "historian": {
        "name": "历史学家",
        "expertise": "历史研究、文化变迁、社会演变",
        "focus": "历史痕迹、时代特征、变迁证据、文化遗产",
    },
    "anthropologist": {
        "name": "人类学家",
        "expertise": "文化研究、社会习俗、人类行为",
        "focus": "文化符号、社会仪式、生活方式、人际互动",
    },
    "urbanist": {
        "name": "城市研究者",
        "expertise": "城市规划、公共空间、城市生活",
        "focus": "空间使用、人流动线、城市肌理、公共与私密",
    },

    # 生活与品味
    "foodie": {
        "name": "美食家",
        "expertise": "烹饪、食材、饮食文化",
        "focus": "食材来源、烹饪技法、风味搭配、饮食文化",
    },
    "craftsman": {
        "name": "工艺师",
        "expertise": "手工艺、制作工艺、材料处理",
        "focus": "工艺细节、制作痕迹、材料处理、匠人精神",
    },
    "storyteller": {
        "name": "故事讲述者",
        "expertise": "叙事、情感、人文关怀",
        "focus": "画面故事、情感细节、人情味、生活瞬间",
    },
}


DIRECTOR_PROMPT = """你是一位智慧的「感知导演」，帮助人们发现日常忽略的细节。

## 你的任务

分析这张图片，选择 **2 个最合适的专业视角** 来观察它（只选2个，不要更多）。

{location_info}

## 可选视角

{perspectives}

## 选择原则

1. **相关性**：选择与图片内容最相关的视角
2. **互补性**：选择的视角应该互补，而非重叠
3. **启发性**：优先选择能带来意外洞见的视角
4. **趣味性**：考虑哪些角度能让普通人觉得"哇，原来如此"

## 示例

- 一朵花 → botanist（植物特性）, artist（色彩美学）, physicist（花瓣结构）
- 天空云彩 → meteorologist（云的类型）, artist（光影变化）, physicist（大气光学）
- 街头小店 → anthropologist（社区文化）, designer（招牌设计）, storyteller（生活故事）
- 一杯咖啡 → foodie（风味）, craftsman（器具工艺）, physicist（萃取原理）

## 输出格式（JSON）

{{
  "image_description": "简述图片内容（1句话），如果能识别出具体地点/建筑请注明",
  "selected_perspectives": ["perspective_id_1", "perspective_id_2"],
  "selection_reason": "为什么选择这些视角（简短说明）"
}}
"""


def director_node(state: dict) -> dict:
    """
    Director 节点：分析图片，选择观察视角

    Args:
        state: 当前状态

    Returns:
        更新后的状态（selected_perspectives）
    """
    print("[Director] 分析图片，选择观察视角...")

    image_data = state["image_data"]
    location_context = state.get("location_context", "")

    # 构建视角列表
    perspectives_text = "\n".join(
        f"- **{pid}** ({p['name']}): {p['expertise']}"
        for pid, p in PERSPECTIVE_POOL.items()
    )

    # 构建位置信息提示
    if location_context:
        location_info = f"""## 位置信息（GPS定位）

{location_context}

请结合位置信息和图片内容进行分析。如果图片中的建筑/地点与附近POI匹配，请在描述中注明。"""
    else:
        location_info = ""

    prompt = DIRECTOR_PROMPT.format(
        perspectives=perspectives_text,
        location_info=location_info,
    )

    try:
        response = call_gemini_with_image(
            prompt=prompt,
            image_data=image_data,
            temperature=0.7,
            max_tokens=1024,
            json_mode=True,
        )

        result = parse_json_response(response, {})

        selected = result.get("selected_perspectives", [])
        description = result.get("image_description", "")
        reason = result.get("selection_reason", "")

        # 验证选择的视角
        valid_perspectives = [p for p in selected if p in PERSPECTIVE_POOL]

        # 至少保证有 2 个视角
        if len(valid_perspectives) < 2:
            valid_perspectives = ["architect", "storyteller"]

        print(f"[Director] 图片: {description}")
        print(f"[Director] 选择视角: {valid_perspectives}")
        print(f"[Director] 原因: {reason}")

        return {
            "image_description": description,
            "selected_perspectives": valid_perspectives,
            "current_phase": "director_complete",
        }

    except Exception as e:
        print(f"[Director] 错误: {e}")
        return {
            "image_description": "",
            "selected_perspectives": ["architect", "storyteller"],
            "current_phase": "director_complete",
            "error": str(e),
        }


def get_perspective_info(perspective_id: str) -> dict:
    """获取视角信息"""
    return PERSPECTIVE_POOL.get(perspective_id, {
        "name": "观察者",
        "expertise": "通用观察",
        "focus": "有趣的细节",
    })
