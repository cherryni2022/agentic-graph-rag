# GLM 模型与 Graphiti 集成问题分析与解决方案

> 日期：2026-01-03  
> 状态：✅ 已解决

---

## 一、问题描述

### 错误现象

在使用 GLM 模型（智谱 AI）调用 Graphiti 构建知识图谱时，在 `extract_attributes_from_node` 阶段出现 Pydantic 验证错误：

```
Error in generating LLM response: 1 validation error for EntityAttributes_45f3656afcad4cf7bbf0e9800714c2db
  Input should be a valid dictionary or instance of EntityAttributes_...
  [type=model_type, input_value=[{'title': 'EntityAttribu...}], input_type=list]
```

### 错误位置

- **文件**: `graphiti_core/utils/maintenance/node_operations.py`
- **方法**: `extract_attributes_from_node` (L364-L413)
- **具体代码** (L394-L413):

```python
unique_model_name = f'EntityAttributes_{uuid4().hex}'
entity_attributes_model = pydantic.create_model(unique_model_name, **attributes_definitions)

llm_response = await llm_client.generate_response(
    prompt_library.extract_nodes.extract_attributes(summary_context),
    response_model=entity_attributes_model,  # 动态创建的 Pydantic 模型
    model_size=ModelSize.small,
)

node.summary = llm_response.get('summary', node.summary)  # ❌ 期望 dict，收到 list
```

---

## 二、根因分析

### 2.1 核心问题

GLM 模型返回的 JSON 是 **列表 (list)** 格式，而 Graphiti 期望的是 **字典 (dict)** 格式。

**GLM 返回**:
```json
[{"title": "EntityAttributes_xxx", "properties": {"summary": {"default": "..."}}}]
```

**期望格式**:
```json
{"summary": "OpenAI is an AI company...", "industry": "Technology"}
```

### 2.2 OpenAI vs GLM 结构化输出对比

| 特性 | OpenAI | GLM (智谱) |
|------|--------|-----------|
| **结构化输出 API** | `beta.chat.completions.parse` 原生支持 | ❌ 不支持 |
| **JSON 模式** | `response_format=response_model` 自动绑定 Schema | `response_format={"type": "json_object"}` 仅保证有效 JSON |
| **Schema 遵循** | 严格遵循 Pydantic 模型结构 | 不保证格式，容易扁平化或数组化 |
| **动态模型支持** | 完美支持 `pydantic.create_model()` | 容易返回错误结构 |

### 2.3 Graphiti 处理流程

```
┌─────────────────────────────────────────────────────────────────┐
│               Graphiti 知识图谱构建流程                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  add_episode()                                                   │
│      │                                                          │
│      ├─► extract_nodes() ─────────────► ✅ 成功                 │
│      │   └─ ExtractedEntities (静态 Pydantic Model)              │
│      │   └─ 之前通过 ZhipuAIClient 已修复                        │
│      │                                                          │
│      ├─► extract_attributes_from_node() ──► ❌ 当前报错位置      │
│      │   └─ pydantic.create_model() 动态创建模型                 │
│      │   └─ GLM 返回 LIST 而不是 DICT                           │
│      │                                                          │
│      ├─► extract_edges() ─────────────► 可能出错                │
│      │                                                          │
│      └─► dedupe_nodes/edges() ────────► 可能出错                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 问题本质

GLM 的 `json_object` 模式只保证输出是**有效的 JSON**，但不保证：
1. JSON 是对象而非数组
2. JSON 结构符合提供的 Schema
3. 字段名和类型与 Pydantic 模型匹配

---

## 三、解决方案

### 3.1 设计理念

不是打补丁式的局部修复，而是设计一个**强健的适配层**，具备：

1. **智能响应规范化** - 自动将各种格式的 LLM 响应转换为期望格式
2. **Schema 简化策略** - 将复杂的动态 Schema 转换为 GLM 更容易理解的格式
3. **多级回退机制** - 当解析失败时，有多种备选策略
4. **响应后处理** - 在验证前对响应进行预处理

### 3.2 架构设计

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ZhipuAIClient V2 架构                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌────────────────┐    ┌──────────────────┐        │
│  │  Prompt +    │ -> │  Schema        │ -> │  GLM API Call    │        │
│  │  Model       │    │  Simplifier    │    │  (json_object)   │        │
│  └──────────────┘    └────────────────┘    └──────────────────┘        │
│                                                    │                     │
│                            ┌───────────────────────┘                     │
│                            ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              ResponseNormalizer 响应规范化层                     │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  1. clean_json_content()    - 移除 markdown 代码块               │   │
│  │  2. normalize_response()    - list→dict 转换                    │   │
│  │  3. extract_fields_from_wrapper() - 提取嵌套字段                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                    │                     │
│                            ┌───────────────────────┘                     │
│                            ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              Multi-Level Fallback Validation 多级回退验证        │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  Level 1: Direct Validation       → 直接 JSON 解析并验证        │   │
│  │  Level 2: Clean + Validate        → 清理后验证                  │   │
│  │  Level 3: Normalize + Validate    → 规范化后验证                │   │
│  │  Level 4: Extract + Validate      → 提取字段后验证              │   │
│  │  Level 5: LLM Retry with Feedback → 带错误反馈重试              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 核心组件

#### ResponseNormalizer - 响应规范化器

```python
class ResponseNormalizer:
    @staticmethod
    def clean_json_content(content: str) -> str:
        """移除 markdown 代码块 (```json...```)"""
        
    @staticmethod
    def normalize_response(raw_response, response_model) -> dict:
        """
        将各种格式规范化为期望的 dict 结构
        - [{}] → {}  (单元素列表转对象)
        - {"properties": {...}} → {...}  (去除 schema 包装)
        """
        
    @staticmethod
    def extract_fields_from_wrapper(response, response_model) -> dict:
        """从 schema 描述格式中提取实际字段值"""
```

#### SchemaSimplifier - Schema 简化器

```python
class SchemaSimplifier:
    @staticmethod
    def simplify_schema(response_model) -> str:
        """生成 GLM 更容易理解的简化 schema 描述"""
        
    @staticmethod
    def create_example(response_model) -> str:
        """创建示例 JSON 输出"""
```

#### 增强的 System Prompt

```python
def _build_system_prompt(self, original_system, response_model):
    enhancement = """
CRITICAL OUTPUT REQUIREMENTS:
1. You MUST output ONLY valid JSON, no explanatory text before or after.
2. Do NOT include markdown code blocks (no ``` markers).
3. The JSON MUST be a single object (dictionary), NOT an array/list.
4. Follow this exact structure:

{simplified_schema}

Full JSON Schema for reference:
{full_schema}

Remember: Output ONLY the JSON object, nothing else.
"""
```

### 3.4 验证与回退策略

```python
def _validate_and_normalize(self, content, response_model):
    # Strategy 1: 直接解析验证
    try:
        parsed = json.loads(content)
        return response_model.model_validate(parsed).model_dump()
    except: pass
    
    # Strategy 2: 清理后验证
    cleaned = self.normalizer.clean_json_content(content)
    try:
        return response_model.model_validate(json.loads(cleaned)).model_dump()
    except: pass
    
    # Strategy 3: 规范化结构后验证
    try:
        normalized = self.normalizer.normalize_response(json.loads(cleaned), response_model)
        return response_model.model_validate(normalized).model_dump()
    except: pass
    
    # Strategy 4: 提取字段后验证
    try:
        extracted = self.normalizer.extract_fields_from_wrapper(normalized, response_model)
        return response_model.model_validate(extracted).model_dump()
    except: pass
    
    # 全部失败 → 抛出异常，触发 LLM 重试
    raise ValidationError(...)
```

---

## 四、测试验证

### 4.1 测试场景

| 测试场景 | 说明 | 状态 |
|---------|------|------|
| `simple_model` | 基础 Pydantic 模型 | ✅ PASSED |
| `static_model` | ExtractedEntities 静态模型 | ✅ PASSED |
| `dynamic_model` | `pydantic.create_model()` 动态模型 (**失败场景**) | ✅ PASSED |

### 4.2 测试输出

```bash
$ python test_zhipu_client.py

Testing Simple Pydantic Model (basic sanity check)
SUCCESS! Response received:
{
  "answer": "Paris",
  "confidence": 1.0
}

Testing Static Pydantic Model (extract_nodes scenario)
SUCCESS! Response received:
{
  "extracted_entities": [
    {"name": "Google", "entity_type_id": 1},
    {"name": "Gemini", "entity_type_id": 3},
    {"name": "DeepMind", "entity_type_id": 1},
    {"name": "Sundar Pichai", "entity_type_id": 2}
  ]
}

Testing Dynamic Pydantic Model (extract_attributes_from_node scenario)
SUCCESS! Response received:
{
  "summary": "OpenAI is an artificial intelligence research laboratory...",
  "industry": "Technology and AI",
  "founded_year": 2015
}

TEST SUMMARY
simple_model: ✅ PASSED
static_model: ✅ PASSED
dynamic_model: ✅ PASSED

🎉 All tests passed! The ZhipuAIClient should now work with Graphiti.
```

---

## 五、文件变更

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `agent/zhipu_llm_client.py` | 重写 | 完整重写，添加 ResponseNormalizer, SchemaSimplifier, 多级回退验证 |
| `test_zhipu_client.py` | 新增 | 测试脚本，覆盖三种模型场景 |
| `TASK.md` | 更新 | 记录 Phase 8b 修复详情 |

---

## 六、使用说明

### 6.1 配置

确保 `.env` 文件包含：

```bash
LLM_PROVIDER=zhipu
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
LLM_API_KEY=your_zhipu_api_key
LLM_CHOICE=glm-4.7
```

### 6.2 自动选择客户端

`graph_utils.py` 会根据模型名称自动选择客户端：

```python
if "glm" in self.llm_choice.lower():
    logger.info(f"Using ZhipuAIClient for GLM model: {self.llm_choice}")
    llm_client = ZhipuAIClient(config=llm_config)
else:
    logger.info(f"Using OpenAIClient for model: {self.llm_choice}")
    llm_client = OpenAIClient(config=llm_config)
```

### 6.3 运行 Ingestion

```bash
source .venv/bin/activate
python -m ingestion.ingest --documents documents --verbose
```

---

## 七、关键经验总结

### 7.1 GLM 结构化输出的局限性

1. `response_format={"type": "json_object"}` **只保证输出是有效 JSON**
2. **不保证** JSON 结构符合提供的 Schema
3. **不保证** 输出是对象而非数组
4. 复杂或动态的 Pydantic 模型容易导致格式错误

### 7.2 解决思路

1. **不要依赖模型自觉遵循 Schema** - 必须有后处理层
2. **多级回退验证** - 一种策略失败时尝试其他策略
3. **简化 Schema 描述** - 帮助模型更好理解期望格式
4. **明确的 Prompt 指令** - 强调输出格式要求

### 7.3 通用适配层设计原则

```
原始响应 → 清理 → 规范化 → 提取 → 验证 → 回退重试
```

这种设计使得客户端能够处理各种不规范的 LLM 输出，提高系统的健壮性。

---

## 八、后续优化建议

1. **监控与日志**: 记录规范化策略命中率，了解 GLM 输出模式
2. **缓存优化**: 对相同 Schema 的简化描述进行缓存
3. **扩展支持**: 将此模式应用到其他非 OpenAI 兼容的模型
4. **性能优化**: 考虑使用更轻量的模型进行辅助格式校正

---

**问题已解决** ✅

修复已提交: `f388617 - feat: Comprehensive GLM structured output fix for dynamic Pydantic models`
