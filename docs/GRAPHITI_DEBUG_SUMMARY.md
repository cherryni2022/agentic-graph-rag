# GLM 模型与 Graphiti 集成：问题分析与解决方案

> **日期**: 2026-01-03  
> **状态**: ✅ 已解决  
> **核心方案**: 放弃 OpenAI 兼容模式，开发专用 ZhipuAIClient

---

## 一、问题背景

### 1.1 项目目标

使用智谱 AI 的 GLM 模型（glm-4.7）替代 OpenAI 模型，通过 Graphiti 框架构建知识图谱。

### 1.2 初始假设

GLM 模型支持 OpenAI 兼容 API，理论上可以直接使用 Graphiti 内置的 `OpenAIClient`。

### 1.3 实际情况

**完全不可行**。GLM 与 OpenAI 在结构化输出方面存在本质差异。

---

## 二、问题分析

### 2.1 第一阶段：基础结构化输出失败

**错误现象**：
```
Error in generating LLM response: Pydantic validation error for ExtractedEntities
```

**根因**：
- OpenAI 使用 `beta.chat.completions.parse` API，原生支持 Pydantic 模型
- GLM 没有等效 API，只能用 `response_format={"type": "json_object"}`
- GLM 返回的 JSON 被包裹在 markdown 代码块中：
  ```
  ```json
  {"extracted_entities": [...]}
  ```
  ```

### 2.2 第二阶段：动态 Pydantic 模型验证失败

在解决了基础问题后，`extract_attributes_from_node` 阶段又出现新错误：

**错误现象**：
```
Input should be a valid dictionary or instance of EntityAttributes_xxx
[type=model_type, input_value=[...], input_type=list]
```

**错误位置**：
- 文件: `graphiti_core/utils/maintenance/node_operations.py`
- 方法: `extract_attributes_from_node` (L394-L413)

**根因**：
- Graphiti 使用 `pydantic.create_model()` 动态创建模型
- GLM 对动态创建的复杂 Schema 理解能力差
- GLM 返回 **列表 (list)** 而非期望的 **字典 (dict)**

**GLM 实际返回**：
```json
[{"title": "EntityAttributes_xxx", "properties": {"summary": {"default": "..."}}}]
```

**期望格式**：
```json
{"summary": "OpenAI is an AI company...", "industry": "Technology"}
```

### 2.3 两种模型的结构化输出对比

| 特性 | OpenAI | GLM (智谱) |
|------|--------|-----------|
| **结构化输出 API** | `beta.chat.completions.parse` | ❌ 不支持 |
| **JSON 模式** | `response_format=response_model` | `response_format={"type": "json_object"}` |
| **Schema 遵循** | 严格遵循 Pydantic 模型 | 仅保证有效 JSON |
| **动态模型支持** | ✅ 完美支持 | ❌ 容易返回错误结构 |
| **输出格式** | 始终是正确的对象 | 可能是列表/嵌套包装 |

---

## 三、解决方案演进

### 3.1 失败尝试一：OpenAI 兼容模式

**方案**：在 `UniversalLLMClient` 中添加 fallback 逻辑

```python
class UniversalLLMClient(OpenAIClient):
    async def _generate_response_with_fallback(self, ...):
        try:
            # 尝试 OpenAI 结构化输出
            return await self.client.beta.chat.completions.parse(...)
        except:
            # 回退到手动解析
            return await self._manual_json_parsing(...)
```

**结果**：❌ 失败
- 每次都触发 fallback，效率低
- 无法处理动态模型的结构问题
- 仍然遇到 list vs dict 问题

### 3.2 失败尝试二：简单 JSON 清理

**方案**：清理 markdown 代码块

```python
def _extract_json(content):
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    return json.loads(content)
```

**结果**：❌ 部分成功
- 解决了 markdown 问题
- 但无法处理 list-to-dict 转换
- 动态模型仍然失败

### 3.3 最终方案：专用 ZhipuAIClient

**核心理念**：放弃让 GLM "假装"是 OpenAI，而是针对 GLM 的特点开发专用客户端。

---

## 四、ZhipuAIClient 架构设计

### 4.1 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ZhipuAIClient 架构                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌────────────────┐    ┌──────────────────┐        │
│  │  Messages +  │ -> │  Schema        │ -> │  GLM API Call    │        │
│  │  Model       │    │  Simplifier    │    │  (json_object)   │        │
│  └──────────────┘    └────────────────┘    └──────────────────┘        │
│         │                    │                      │                    │
│         │  增强 Prompt       │ 简化 Schema          │                    │
│         │                    │                      │                    │
│         ▼                    ▼                      ▼                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              ResponseNormalizer 响应规范化层                     │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  clean_json_content()      → 移除 markdown 代码块                │   │
│  │  normalize_response()      → list→dict 转换                     │   │
│  │  extract_fields_from_wrapper() → 提取嵌套字段                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                    │                                     │
│                                    ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              Multi-Level Fallback Validation                     │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  Level 1: Direct Validation       → 直接解析验证                 │   │
│  │  Level 2: Clean + Validate        → 清理后验证                   │   │
│  │  Level 3: Normalize + Validate    → 规范化后验证                 │   │
│  │  Level 4: Extract + Validate      → 提取字段后验证               │   │
│  │  Level 5: LLM Retry               → 带错误反馈重试               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 核心组件

#### ResponseNormalizer

处理 GLM 响应的各种不规范格式：

```python
class ResponseNormalizer:
    @staticmethod
    def clean_json_content(content: str) -> str:
        """移除 markdown 代码块"""
        if content.startswith('```json'):
            content = content[7:]
        if content.endswith('```'):
            content = content[:-3]
        return content.strip()
    
    @staticmethod
    def normalize_response(raw_response, response_model) -> dict:
        """
        将不规范的响应转换为期望格式
        - [{}] → {}  (单元素列表转对象)
        - {"properties": {...}} → {...}  (去除 schema 包装)
        """
        if isinstance(raw_response, list) and len(raw_response) == 1:
            return ResponseNormalizer.normalize_response(raw_response[0], response_model)
        return raw_response
    
    @staticmethod
    def extract_fields_from_wrapper(response, response_model) -> dict:
        """从 schema 描述格式中提取实际字段值"""
        # 处理 {"field": {"default": "value"}} → {"field": "value"}
        ...
```

#### SchemaSimplifier

将复杂的 Pydantic Schema 转换为 GLM 更容易理解的格式：

```python
class SchemaSimplifier:
    @staticmethod
    def simplify_schema(response_model) -> str:
        """生成简化的 schema 描述"""
        return """{
          "summary": string (required) - Entity summary
          "industry": string (optional) - Industry category
        }"""
```

#### 增强的 System Prompt

明确告知 GLM 输出要求：

```python
def _build_system_prompt(self, original_system, response_model):
    return f"""{original_system}

CRITICAL OUTPUT REQUIREMENTS:
1. Output ONLY valid JSON, no explanatory text.
2. Do NOT include markdown code blocks.
3. The JSON MUST be a single object (dict), NOT an array (list).
4. Follow this structure:

{simplified_schema}

Full JSON Schema:
{full_schema}
"""
```

### 4.3 多级回退验证

```python
def _validate_and_normalize(self, content, response_model):
    # Strategy 1: 直接解析
    try:
        return response_model.model_validate(json.loads(content)).model_dump()
    except: pass
    
    # Strategy 2: 清理后解析
    cleaned = self.normalizer.clean_json_content(content)
    try:
        return response_model.model_validate(json.loads(cleaned)).model_dump()
    except: pass
    
    # Strategy 3: 规范化结构
    normalized = self.normalizer.normalize_response(json.loads(cleaned), response_model)
    try:
        return response_model.model_validate(normalized).model_dump()
    except: pass
    
    # Strategy 4: 提取嵌套字段
    extracted = self.normalizer.extract_fields_from_wrapper(normalized, response_model)
    return response_model.model_validate(extracted).model_dump()
```

---

## 五、集成方式

### 5.1 自动客户端选择

在 `graph_utils.py` 中根据模型名称自动选择客户端：

```python
from .zhipu_llm_client import ZhipuAIClient
from graphiti_core.llm_client.openai_client import OpenAIClient

# 根据模型自动选择客户端
if "glm" in self.llm_choice.lower():
    logger.info(f"Using ZhipuAIClient for GLM model: {self.llm_choice}")
    llm_client = ZhipuAIClient(config=llm_config)
else:
    logger.info(f"Using OpenAIClient for model: {self.llm_choice}")
    llm_client = OpenAIClient(config=llm_config)
```

### 5.2 配置

`.env` 文件：
```bash
LLM_PROVIDER=zhipu
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
LLM_API_KEY=your_zhipu_api_key
LLM_CHOICE=glm-4.7

# 降低并发以避免速率限制
SEMAPHORE_LIMIT=5
```

---

## 六、关键经验总结

### 6.1 不要假设 API 兼容性

> "支持 OpenAI API 格式" ≠ "功能完全等同"

GLM 虽然 API 格式兼容 OpenAI，但在结构化输出等高级功能上存在本质差异。

### 6.2 为不同模型开发专用客户端

参考 Graphiti 自身的设计：
- `OpenAIClient` - 针对 OpenAI 优化
- `GeminiClient` - 针对 Google Gemini 优化
- `ZhipuAIClient` - 针对智谱 GLM 优化

每个客户端了解其目标模型的特点和局限性。

### 6.3 多级回退是必要的

不能假设 LLM 总是返回正确格式。需要：
1. 多种解析策略
2. 智能错误恢复
3. 带上下文的重试机制

### 6.4 增强 Prompt 而非依赖 API

对于不支持原生结构化输出的模型，通过以下方式提高成功率：
- 在 system prompt 中明确输出格式要求
- 提供简化的 schema 描述
- 给出示例输出

---

## 七、文件变更清单

| 文件 | 变更 | 说明 |
|------|------|------|
| `agent/zhipu_llm_client.py` | 新增/重写 | 专用 ZhipuAI 客户端 |
| `agent/graph_utils.py` | 修改 | 添加客户端自动选择逻辑 |
| `requirements.txt` | 修改 | 添加 `zhipuai>=2.1.0` |
| `test_zhipu_client.py` | 新增 | 客户端测试脚本 |

---

## 八、结论

### 问题

GLM 模型无法通过 OpenAI 兼容模式可靠地在 Graphiti 中使用，特别是：
1. 不支持 `beta.chat.completions.parse` 结构化输出 API
2. `json_object` 模式不保证 Schema 遵循
3. 动态 Pydantic 模型导致格式错误

### 解决方案

开发专用 `ZhipuAIClient`，具备：
- **ResponseNormalizer**: 智能响应规范化
- **SchemaSimplifier**: Schema 简化与描述增强
- **多级回退验证**: 多种解析策略保证成功率
- **增强 Prompt**: 明确输出格式要求

### 结果

✅ GLM 模型现在可以可靠地在 Graphiti 中构建知识图谱

---

*文档更新日期: 2026-01-03*
