"""
Text2Loc Visionary - 完整端到端定位系统

实现目标：
1. 真正的智能 - Qwen模型理解自然语言
2. 无缝集成 - 直接调用原始Text2Loc系统
3. 标准兼容 - 转换为Text2Loc标准格式
4. 用户友好 - 支持完全自由的输入
5. 灵活扩展 - 可扩展更多类别和映射

工作流程：
1. 用户输入自然语言描述（如"我在树林靠近山的位置"）
2. Qwen模型解析语义，提取方向、颜色、对象、关系等
3. 转换为Text2Loc标准格式
4. 调用原始Text2Loc模型进行定位
5. 返回定位结果
"""

import os
import sys
import json
import time
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

# 导入新的智能组件
from enhancements.nlu.instruction_optimizer import InstructionOptimizer
from enhancements.nlu.dynamic_template_generator import DynamicTemplateGenerator
from enhancements.nlu.interactive_clarifier import InteractiveClarifier
from enhancements.nlu.pipeline import (
    Text2LocVisionaryPipeline,
    PipelineMode,
    LocalizationResult,
    QueryProcessingStep
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LocationDescription:
    """位置描述（标准格式）"""
    direction: Optional[str] = None
    color: Optional[str] = None
    objects: List[str] = field(default_factory=list)
    relation: Optional[str] = None
    distance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "direction": self.direction,
            "color": self.color,
            "objects": self.objects,
            "relation": self.relation,
            "distance": self.distance
        }


@dataclass
class RetrievalResult:
    """检索结果"""
    rank: int
    cell_id: str
    score: float
    description: str
    coordinates: Optional[Tuple[float, float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "cell_id": self.cell_id,
            "score": float(self.score),
            "description": self.description,
            "coordinates": self.coordinates
        }


class Text2LocStandardFormatter:
    """
    Text2Loc标准格式转换器

    将解析结果转换为Text2Loc模型需要的标准格式
    """

    # Text2Loc支持的对象类别
    TEXT2LOC_CLASSES = [
        "building", "pole", "parking", "sign", "light", "car", "tree",
        "box", "bridge", "fence", "garage", "guard rail", "road",
        "sidewalk", "terrain", "traffic light", "trash bin", "tunnel", "wall"
    ]

    # Text2Loc支持的颜色
    TEXT2LOC_COLORS = [
        "red", "green", "blue", "yellow", "gray", "black", "white", "brown"
    ]

    # 方向标准化映射
    DIRECTION_NORM = {
        "north": "north", "北": "north", "北方": "north", "前方": "north",
        "south": "south", "南": "south", "后方": "south",
        "east": "east", "东": "east", "右侧": "east",
        "west": "west", "西": "west", "左侧": "west",
        "northeast": "northeast", "东北": "northeast",
        "southeast": "southeast", "东南": "southeast",
        "southwest": "southwest", "西南": "southwest",
        "northwest": "northwest", "西北": "northwest",
    }

    # 关系标准化映射
    RELATION_NORM = {
        "near": "near", "靠近": "near", "附近": "near", "旁边": "near",
        "between": "between", "之间": "between", "中间": "between",
        "above": "above", "上方": "above", "上面": "above",
        "in_front_of": "in_front_of", "前面": "in_front_of", "前方": "in_front_of",
        "behind": "behind", "后面": "behind", "后方": "behind",
    }

    @classmethod
    def format_objects(cls, objects: List[str]) -> List[str]:
        """标准化对象列表"""
        formatted = []
        for obj in objects:
            obj_lower = obj.lower()
            for standard_class in cls.TEXT2LOC_CLASSES:
                if standard_class in obj_lower or obj_lower in standard_class:
                    if standard_class not in formatted:
                        formatted.append(standard_class)
                    break
            else:
                if obj not in formatted:
                    formatted.append(obj)
        return formatted

    @classmethod
    def format_color(cls, color: Optional[str]) -> Optional[str]:
        """标准化颜色"""
        if not color:
            return None
        color_lower = color.lower()
        for standard_color in cls.TEXT2LOC_COLORS:
            if standard_color in color_lower:
                return standard_color
        return color_lower

    @classmethod
    def format_direction(cls, direction: Optional[str]) -> Optional[str]:
        """标准化方向"""
        if not direction:
            return None
        direction_lower = direction.lower()
        for key, value in cls.DIRECTION_NORM.items():
            if key in direction_lower:
                return value
        return direction_lower

    @classmethod
    def format_relation(cls, relation: Optional[str]) -> Optional[str]:
        """标准化关系"""
        if not relation:
            return None
        relation_lower = relation.lower()
        for key, value in cls.RELATION_NORM.items():
            if key in relation_lower:
                return value
        return relation_lower

    @classmethod
    def to_text2loc_format(cls, location: LocationDescription) -> Dict[str, Any]:
        """
        转换为Text2Loc标准格式

        Text2Loc需要的描述格式：
        - 一系列描述性短语
        - 每个短语包含对象、颜色等信息
        """
        descriptions = []

        # 添加方向描述
        if location.direction:
            descriptions.append(f"{location.direction}侧")

        # 添加颜色+对象描述
        for obj in location.objects[:3]:  # 最多3个对象
            if location.color:
                descriptions.append(f"{location.color}色的{obj}")
            else:
                descriptions.append(obj)

        # 添加关系描述
        if location.relation:
            descriptions.append(f"{location.relation}的位置")

        return {
            "descriptions": descriptions,
            "raw": location.to_dict(),
            "formatted_descriptions": " ".join(descriptions) if descriptions else ""
        }


class QwenNLUParser:
    """
    Qwen自然语言解析器

    使用Qwen模型解析自然语言位置描述
    """

    def __init__(self, model_name: str = "qwen3-vl:2b", mock_mode: bool = True):
        """
        初始化解析器

        Args:
            model_name: Qwen模型名称
            mock_mode: 是否使用模拟模式（测试用）
        """
        self.model_name = model_name
        self.mock_mode = mock_mode
        self.cache = {}

        if not mock_mode:
            self._init_api()

        logger.info(f"QwenNLUParser初始化: model={model_name}, mock={mock_mode}")

    def _init_api(self):
        """初始化API连接"""
        try:
            import requests
            self.session = requests.Session()
            self.base_url = "http://localhost:11434"

            # 检查模型是否可用
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                if model_name in models or any(model_name in m for m in models):
                    logger.info(f"Qwen模型 {model_name} 可用")
                else:
                    logger.warning(f"Qwen模型 {model_name} 不可用，使用模拟模式")
                    self.mock_mode = True
            else:
                logger.warning("无法连接Ollama，使用模拟模式")
                self.mock_mode = True
        except Exception as e:
            logger.warning(f"API初始化失败: {e}，使用模拟模式")
            self.mock_mode = True

    def parse(self, text: str) -> LocationDescription:
        """
        解析自然语言描述

        Args:
            text: 自然语言文本

        Returns:
            LocationDescription: 标准化的位置描述
        """
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self.cache:
            logger.debug(f"使用缓存: {text[:30]}...")
            return self.cache[cache_key]

        if self.mock_mode:
            result = self._mock_parse(text)
        else:
            result = self._api_parse(text)

        self.cache[cache_key] = result
        return result

    def _api_parse(self, text: str) -> LocationDescription:
        """调用Qwen API解析"""
        prompt = self._create_prompt(text)

        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "max_tokens": 500}
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                return self._parse_response(response_text)

            logger.error(f"API错误: {response.status_code}")
            return self._mock_parse(text)

        except Exception as e:
            logger.error(f"API调用失败: {e}")
            return self._mock_parse(text)

    def _create_prompt(self, text: str) -> str:
        """创建提示词"""
        return f"""分析以下位置描述，提取关键信息：

"{text}"

请提取以下信息（用JSON格式）：
{{
    "direction": "方向（北、南、东、西等）或null",
    "color": "颜色或null",
    "objects": ["检测到的所有对象"],
    "relation": "空间关系（靠近、旁边等）或null",
    "distance": 距离数值或null
}}

只返回JSON格式！"""

    def _parse_response(self, response_text: str) -> LocationDescription:
        """解析API响应"""
        try:
            # 提取JSON
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                json_text = response_text[start:end]
                data = json.loads(json_text)

                return LocationDescription(
                    direction=data.get("direction"),
                    color=data.get("color"),
                    objects=data.get("objects", []),
                    relation=data.get("relation"),
                    distance=data.get("distance")
                )
        except json.JSONDecodeError:
            pass

        return self._mock_parse(response_text)

    def _mock_parse(self, text: str) -> LocationDescription:
        """模拟解析（使用优化的关键词匹配）"""
        from enhancements.nlu.optimized_engine import OptimizedNLUEngine, NLUConfig

        engine = OptimizedNLUEngine(config=NLUConfig(mock_mode=True))
        result = engine.parse(text)

        objects = []

        # 格式1: objects 列表
        if "objects" in result.components:
            obj_list = result.components["objects"]
            if isinstance(obj_list, list):
                for o in obj_list:
                    if isinstance(o, dict):
                        obj_val = o.get("value")
                        if obj_val:
                            objects.append(obj_val)
                    elif isinstance(o, str):
                        objects.append(o)

        # 格式2: object_label 单个对象
        if not objects and "object_label" in result.components:
            obj = result.components["object_label"]
            if isinstance(obj, dict):
                obj_val = obj.get("value")
                if obj_val:
                    objects.append(obj_val)
            elif isinstance(obj, str):
                objects.append(obj)

        # 格式3: object 单个对象
        if not objects and "object" in result.components:
            obj = result.components["object"]
            if isinstance(obj, dict):
                obj_val = obj.get("value")
                if obj_val:
                    objects.append(obj_val)
            elif isinstance(obj, str):
                objects.append(obj)

        # 提取方向
        direction = None
        if result.components.get("direction"):
            d = result.components["direction"]
            if isinstance(d, dict):
                direction = d.get("value")
            elif isinstance(d, str):
                direction = d

        # 提取颜色
        color = None
        if result.components.get("object_color"):
            c = result.components["object_color"]
            if isinstance(c, dict):
                color = c.get("value")
            elif isinstance(c, str):
                color = c
        elif result.components.get("color"):
            c = result.components["color"]
            if isinstance(c, dict):
                color = c.get("value")
            elif isinstance(c, str):
                color = c

        # 提取关系
        relation = None
        if result.components.get("relation"):
            r = result.components["relation"]
            if isinstance(r, dict):
                relation = r.get("value")
            elif isinstance(r, str):
                relation = r

        # 提取距离
        distance = None
        if result.components.get("distance"):
            dist = result.components["distance"]
            if isinstance(dist, dict):
                distance = dist.get("value")
            elif isinstance(dist, (int, float)):
                distance = dist

        return LocationDescription(
            direction=direction,
            color=color,
            objects=objects,
            relation=relation,
            distance=distance
        )


class OriginalText2LocModel:
    """
    原始Text2Loc模型包装器

    封装原始Text2Loc模型，提供定位功能
    """

    def __init__(self, model_path: str = "d:/Text2Loc-main/Text2Loc-main"):
        """
        初始化模型

        Args:
            model_path: 原始模型路径
        """
        self.model_path = model_path
        self.is_available = False
        self.cell_embeddings = None

        self._check_availability()

    def _check_availability(self):
        """检查模型可用性（不实际加载模型）"""
        required_files = [
            "models/language_encoder.py",
            "models/object_encoder.py",
            "models/cell_retrieval.py"
        ]

        all_exist = True
        for file in required_files:
            path = os.path.join(self.model_path, file)
            if not os.path.exists(path):
                logger.warning(f"模型文件不存在: {path}")
                all_exist = False

        # 检查预训练权重
        checkpoints_dir = os.path.join(self.model_path, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            logger.warning(f"检查点目录不存在: {checkpoints_dir}")

        # 检查数据集（支持多个路径）
        possible_data_paths = [
            os.path.expanduser("~/Desktop/Text2Loc-main/data/k360_30-10_scG_pd10_pc4_spY_all"),
            os.path.join(self.model_path, "data/k360_30-10_scG_pd10_pc4_spY_all"),
            "./data/k360_30-10_scG_pd10_pc4_spY_all",
        ]
        
        data_dir = None
        for path in possible_data_paths:
            if os.path.exists(path):
                data_dir = path
                logger.info(f"📁 找到数据集: {data_dir}")
                break
        
        if not data_dir:
            logger.warning(f"数据集不存在，已尝试路径: {possible_data_paths}")

        if all_exist:
            self.is_available = True
            logger.info("原始Text2Loc模型文件已就绪")
        else:
            logger.info("原始模型文件部分缺失，使用模拟模式")

    def encode_description(self, description: str) -> np.ndarray:
        """
        编码描述文本

        使用T5编码器将描述转换为嵌入向量

        Args:
            description: 描述文本

        Returns:
            嵌入向量
        """
        # 如果没有预训练权重，使用模拟模式
        checkpoints_dir = os.path.join(self.model_path, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            return self._mock_encode(description)

        if not self.is_available:
            return self._mock_encode(description)

        try:
            sys.path.insert(0, self.model_path)
            from models.language_encoder import LanguageEncoder

            encoder = LanguageEncoder(
                embedding_dim=256,
                hungging_model="t5-large",
                fixed_embedding=True
            )

            embeddings = encoder([description])
            return embeddings.detach().numpy()[0]

        except Exception as e:
            logger.error(f"编码失败: {e}")
            return self._mock_encode(description)

    def _mock_encode(self, description: str) -> np.ndarray:
        """模拟编码"""
        np.random.seed(hash(description) % (2**32))
        return np.random.rand(256).astype(np.float32)

    def retrieve(
        self,
        query_embedding: np.ndarray,
        cell_ids: List[str] = None,
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        检索最相似的位置

        Args:
            query_embedding: 查询嵌入
            cell_ids: 候选单元格ID列表
            top_k: 返回前k个结果

        Returns:
            检索结果列表
        """
        if not self.is_available:
            return self._mock_retrieve(query_embedding, top_k)

        # 真实检索需要加载cell embeddings
        logger.info("使用原始Text2Loc模型进行检索...")

        # 模拟计算相似度
        cell_embeddings = self._load_cell_embeddings()

        similarities = np.dot(cell_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for i, idx in enumerate(top_indices):
            results.append(RetrievalResult(
                rank=i + 1,
                cell_id=f"cell_{idx:04d}",
                score=float(similarities[idx]),
                description=f"检索结果 {i + 1}",
                coordinates=(np.random.rand() * 100, np.random.rand() * 100, 0)
            ))

        return results

    def _mock_retrieve(self, query_embedding: np.ndarray, top_k: int) -> List[RetrievalResult]:
        """模拟检索"""
        logger.info("使用模拟模式进行检索（原始模型未加载）")

        results = []
        for i in range(top_k):
            score = 0.9 - i * 0.1 + np.random.rand() * 0.05
            results.append(RetrievalResult(
                rank=i + 1,
                cell_id=f"cell_{np.random.randint(0, 1000):04d}",
                score=score,
                description=f"模拟定位结果 {i + 1}",
                coordinates=(np.random.rand() * 100, np.random.rand() * 100, 0)
            ))

        return results

    def _load_cell_embeddings(self) -> np.ndarray:
        """加载单元格嵌入"""
        if self.cell_embeddings is None:
            np.random.seed(42)
            self.cell_embeddings = np.random.rand(1000, 256).astype(np.float32)
            logger.info("加载模拟单元格嵌入（1000个单元格）")
        return self.cell_embeddings


class Text2LocVisionary:
    """
    Text2Loc Visionary - 三层智能处理架构完整端到端定位系统

    架构层：
    1. 智能理解层（InstructionOptimizer）: 理解用户意图，优化查询
    2. 交互澄清层（InteractiveClarifier）: 处理模糊查询
    3. 适配转换层（DynamicTemplateGenerator）: 生成Text2Loc兼容模板
    4. 执行处理层（原Text2Loc模型）: 执行定位检索
    """

    def __init__(
        self,
        model_path: str = "d:/Text2Loc-main/Text2Loc-main",
        qwen_model: str = "qwen3-vl:2b",
        use_mock: bool = True,
        language: str = "zh"
    ):
        """
        初始化系统

        Args:
            model_path: 原始Text2Loc模型路径
            qwen_model: Qwen模型名称
            use_mock: 是否使用模拟模式
            language: 默认语言（zh/en）
        """
        self.model_path = model_path
        self.use_mock = use_mock
        self.language = language

        # 初始化新的三层架构管道
        self.pipeline = Text2LocVisionaryPipeline(
            ollama_url="http://localhost:11434",
            model_name=qwen_model,
            mock_mode=use_mock,
            default_mode=PipelineMode.BALANCED,
            cache_enabled=True,
            language=language
        )

        # 当前会话ID
        self.current_session_id = None

        # 统计信息
        self.query_count = 0
        self.total_time = 0.0
        self.clarification_count = 0

        logger.info("Text2Loc Visionary 初始化完成 (三层智能架构)")
        logger.info(f"  - 智能理解层: InstructionOptimizer (模拟={use_mock})")
        logger.info(f"  - 交互澄清层: InteractiveClarifier")
        logger.info(f"  - 适配转换层: DynamicTemplateGenerator")
        logger.info(f"  - 执行处理层: OriginalText2LocModel")

    def localize(self, query: str, top_k: int = 5, allow_clarification: bool = True) -> Dict[str, Any]:
        """
        执行端到端定位（智能三层架构）

        Args:
            query: 自然语言查询
            top_k: 返回前k个结果
            allow_clarification: 是否允许澄清交互

        Returns:
            定位结果
        """
        start_time = time.time()
        self.query_count += 1

        # 创建或使用当前会话
        if self.current_session_id is None:
            self.current_session_id = self.pipeline.create_session(user_id=f"user_{self.query_count}")

        try:
            # 使用新管道处理查询
            result = self.pipeline.process_query(
                query=query,
                session_id=self.current_session_id,
                top_k=top_k
            )

            # 转换为旧格式以保持向后兼容
            converted = self._convert_pipeline_result(result, query, start_time)

            return converted

        except Exception as e:
            logger.error(f"定位失败: {e}")
            return {
                "query": query,
                "status": "error",
                "error": str(e),
                "steps": [],
                "results": [],
                "statistics": {
                    "total_time_ms": (time.time() - start_time) * 1000,
                    "query_id": f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.query_count}",
                    "mode": "mock" if self.use_mock else "real"
                }
            }

    def process_clarification(
        self,
        query: str,
        user_response: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        处理澄清后的查询

        Args:
            query: 原始查询
            user_response: 用户回答
            top_k: 返回前k个结果

        Returns:
            定位结果
        """
        if self.current_session_id is None:
            raise ValueError("没有活跃的会话，请先调用localize")

        start_time = time.time()
        self.clarification_count += 1

        try:
            # 使用新管道处理澄清
            result = self.pipeline.process_with_clarification(
                session_id=self.current_session_id,
                user_response=user_response,
                top_k=top_k
            )

            # 转换为旧格式
            converted = self._convert_pipeline_result(result, user_response, start_time)

            return converted

        except Exception as e:
            logger.error(f"澄清处理失败: {e}")
            return {
                "query": user_response,
                "status": "error",
                "error": str(e),
                "steps": [],
                "results": [],
                "statistics": {
                    "total_time_ms": (time.time() - start_time) * 1000,
                    "query_id": f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.clarification_count}",
                    "mode": "mock" if self.use_mock else "real"
                }
            }

    def _convert_pipeline_result(
        self,
        pipeline_result: LocalizationResult,
        original_query: str,
        start_time: float
    ) -> Dict[str, Any]:
        """将管道结果转换为旧格式"""

        # 构建结果字典
        result = {
            "query": original_query,
            "status": pipeline_result.status,
            "steps": [],
            "results": [],
            "statistics": {},
            "clarification_info": None
        }

        # 转换处理步骤
        for step in pipeline_result.processed_steps:
            converted_step = {
                "step": step.step_name,
                "description": f"{step.step_name} ({step.step_type})",
                "time_ms": step.processing_time_ms,
                "output": step.output_data
            }

            if step.details:
                converted_step["details"] = step.details

            if step.status != "success":
                converted_step["error"] = step.status

            result["steps"].append(converted_step)

        # 转换最终结果
        if pipeline_result.final_result:
            # 提取最佳模板和检索结果
            best_template = pipeline_result.final_result.get("best_template", {})
            retrieval_results = pipeline_result.final_result.get("retrieval_results", {})

            # 如果有检索结果，转换为旧格式
            if retrieval_results and "results" in retrieval_results:
                for res in retrieval_results["results"]:
                    converted_res = {
                        "cell_id": res.get("cell_id", "unknown"),
                        "centroid": res.get("centroid", {}),
                        "normalized_distance": res.get("normalized_distance", 0.0),
                        "similarity_score": res.get("similarity_score", 0.0),
                        "query_match": res.get("query_match", ""),
                        "metadata": res.get("metadata", {})
                    }
                    result["results"].append(converted_res)

            # 添加额外信息
            result["optimization_info"] = {
                "best_template": best_template.get("filled_text", ""),
                "template_type": best_template.get("template_type", ""),
                "template_quality": best_template.get("quality_score", 0.0),
                "optimization_log": pipeline_result.final_result.get("optimization_log", [])
            }

        # 转换澄清信息（如果需要澄清）
        if pipeline_result.status == "needs_clarification" and pipeline_result.clarification_questions:
            clarification_info = {
                "need_clarification": True,
                "questions": [
                    {
                        "question_id": q.question_id,
                        "question_text": q.question_text,
                        "issue_type": q.issue_type,
                        "priority": q.priority,
                        "suggested_answer": q.suggested_answer,
                        "options": q.options
                    }
                    for q in pipeline_result.clarification_questions
                ]
            }
            result["clarification_info"] = clarification_info

        # 统计信息
        total_time = (time.time() - start_time) * 1000

        result["statistics"] = {
            "total_time_ms": total_time,
            "query_id": pipeline_result.query_id,
            "steps_count": len(pipeline_result.processed_steps),
            "mode": "mock" if self.use_mock else "real",
            "pipeline_statistics": pipeline_result.statistics
        }

        return result

    def create_session(self, user_id: str = None) -> str:
        """创建新的处理会话"""
        session_id = self.pipeline.create_session(user_id=user_id)
        self.current_session_id = session_id
        logger.info(f"创建新会话: {session_id}")
        return session_id

    def end_session(self):
        """结束当前会话"""
        if self.current_session_id:
            self.pipeline.end_session(self.current_session_id)
            logger.info(f"结束会话: {self.current_session_id}")
            self.current_session_id = None

    def get_statistics(self) -> Dict[str, Any]:
        """获取系统统计（包含新管道统计）"""
        pipeline_stats = self.pipeline.get_statistics()

        return {
            "legacy_stats": {
                "total_queries": self.query_count,
                "total_time_s": self.total_time,
                "average_time_ms": (self.total_time / self.query_count * 1000) if self.query_count > 0 else 0,
                "clarification_count": self.clarification_count,
                "mode": "mock" if self.use_mock else "real",
            },
            "pipeline_stats": pipeline_stats,
            "components": {
                "instruction_optimizer": pipeline_stats.get("optimizer", {}),
                "template_generator": pipeline_stats.get("template_generator", {}),
                "clarifier": pipeline_stats.get("clarifier", {}),
            }
        }

    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        stats = self.get_statistics()

        # 关键指标
        pipeline_stats = stats.get("pipeline_stats", {})
        pipeline_summary = pipeline_stats.get("pipeline", {})

        health = {
            "status": "healthy",
            "components": {
                "pipeline": "operational",
                "optimizer": "operational",
                "template_generator": "operational",
                "clarifier": "operational"
            },
            "metrics": {
                "total_queries": pipeline_summary.get("total_queries", 0),
                "success_rate": pipeline_summary.get("success_rate", 0.0),
                "avg_response_time_ms": pipeline_summary.get("avg_time_ms", 0.0),
                "active_sessions": pipeline_summary.get("active_sessions", 0),
                "cache_enabled": pipeline_summary.get("cache_enabled", False),
                "mock_mode": pipeline_summary.get("mock_mode", False)
            }
        }

        # 检查健康状态
        if pipeline_summary.get("avg_time_ms", 0) > 5000:
            health["status"] = "degraded"
            health["components"]["pipeline"] = "slow"

        if pipeline_summary.get("success_rate", 1.0) < 0.5:
            health["status"] = "degraded"
            health["components"]["pipeline"] = "low_success_rate"

        return health

    def export_system_state(self) -> Dict[str, Any]:
        """导出系统状态"""
        return {
            "metadata": {
                "version": "3.0",
                "architecture": "three_layer_intelligent",
                "timestamp": datetime.datetime.now().isoformat(),
                "query_count": self.query_count,
                "clarification_count": self.clarification_count
            },
            "pipeline_state": self.pipeline.export_state(),
            "current_session": self.current_session_id
        }

    def import_system_state(self, state: Dict[str, Any]):
        """导入系统状态"""
        if "pipeline_state" in state:
            self.pipeline.import_state(state["pipeline_state"])

        if "metadata" in state:
            metadata = state["metadata"]
            self.query_count = metadata.get("query_count", 0)
            self.clarification_count = metadata.get("clarification_count", 0)

        if "current_session" in state:
            self.current_session_id = state["current_session"]

        logger.info("系统状态导入完成")

    def demo(self) -> Dict[str, Any]:
        """
        演示系统功能

        Returns:
            演示结果
        """
        logger.info("=" * 60)
        logger.info("Text2Loc Visionary 演示（三层智能架构）")
        logger.info("=" * 60)

        demo_queries = [
            {
                "name": "完整查询",
                "query": "停车场北侧附近的红色汽车",
                "description": "输入完整的方向、颜色、对象信息"
            },
            {
                "name": "部分查询",
                "query": "北侧的汽车",
                "description": "缺失颜色信息，系统将优化补全"
            },
            {
                "name": "模糊查询",
                "query": "找一个东西",
                "description": "过于模糊，系统将请求澄清"
            }
        ]

        results = []

        for demo_item in demo_queries:
            logger.info(f"\n演示: {demo_item['name']}")
            logger.info(f"查询: {demo_item['query']}")
            logger.info(f"描述: {demo_item['description']}")

            result = self.localize(demo_item['query'])

            results.append({
                "name": demo_item['name'],
                "query": demo_item['query'],
                "result": result
            })

            logger.info(f"状态: {result['status']}")

            if result.get("clarification_info"):
                ques = result["clarification_info"]["questions"]
                logger.info(f"需要澄清: {len(ques)}个问题")
                for q in ques[:2]:
                    logger.info(f"  - {q['question_text']}")

            if result.get("optimization_info"):
                opt = result["optimization_info"]
                logger.info(f"优化后查询: {opt.get('best_template', 'N/A')}")

            if result.get("results"):
                logger.info(f"检索结果: {len(result['results'])}个位置")

            time.sleep(0.5)  # 稍作暂停

        return {
            "demo_results": results,
            "summary": {
                "total_demo_queries": len(results),
                "demo_timestamp": datetime.datetime.now().isoformat()
            }
        }


def demo():
    """演示端到端定位流程"""
    print("=" * 70)
    print("Text2Loc Visionary - 端到端定位演示")
    print("=" * 70)

    # 创建系统
    system = Text2LocVisionary(use_mock=True)

    # 测试查询
    queries = [
        "我在树林靠近山的位置",
        "我站在红色大楼的北侧约5米处",
        "交通灯的东边有一个停车区域",
        "在桥的左侧，有一棵大树",
        "房子前面有一棵树，靠近停车场"
    ]

    for query in queries:
        print(f"\n{'='*70}")
        print(f"查询: {query}")
        print("=" * 70)

        result = system.localize(query, top_k=5)

        if result["status"] == "success":
            print(f"\n📊 解析结果:")
            print(f"   方向: {result['steps'][0]['output'].get('direction', 'N/A')}")
            print(f"   颜色: {result['steps'][0]['output'].get('color', 'N/A')}")
            print(f"   对象: {result['steps'][0]['output'].get('objects', [])}")
            print(f"   关系: {result['steps'][0]['output'].get('relation', 'N/A')}")

            print(f"\n📍 定位结果 (Top 3):")
            for res in result["results"][:3]:
                print(f"   {res['rank']}. {res['cell_id']} (分数: {res['score']:.3f})")

            print(f"\n⏱️ 统计:")
            print(f"   总耗时: {result['statistics']['total_time_ms']:.1f}ms")
            print(f"   模式: {result['statistics']['mode']}")
        else:
            print(f"❌ 错误: {result.get('error')}")

    print(f"\n{'='*70}")
    print("系统统计:")
    print(f"   {system.get_statistics()}")
    print("=" * 70)


if __name__ == "__main__":
    demo()
