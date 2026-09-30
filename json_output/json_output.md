# 大模型输出 JSON

## 测试维度

| 维度 | 典型失败 | 对应任务 |
| --- | --- | --- |
| 语法 | 代码围栏、解释文字、截断、空内容 | 所有任务 |
| 结构 | 漏字段、错类型、多字段、嵌套结构错误 | 所有任务 |
| 内容 | JSON 和 schema 都正确，但值错 | job_posting、person |
| 多步推理 | 折扣算错，或条件优先级用错 | invoice_reasoning、rule_priority |
| 输入隔离 | 把待抽取文本里的伪指令当成命令 | instruction_in_data |

CoT（思考模式）是实验条件，不是独立的 JSON 字段。用同一批任务比较开启和关闭思考模式，只对最终 content 的 JSON 判分。DeepSeek 把思考内容放在单独的 reasoning_content 字段；脚本只记录其长度。

先运行每组 1 轮检查调用和评分，再增加轮数观察波动：

    uv run json_output/json_output.py 1 --compare-thinking
    uv run json_output/json_output.py 5 --compare-thinking

这会得到 prompt_only/json_object 与 no_thinking/thinking 的四组对照。一次变好或变差不足以证明思考模式的效果。如果 finish_reason 为 length，应先调大 MAX_TOKENS 再测。

## 通过prompt约束

```python
TEXT = "张三今年27岁，和老婆生了个女儿"
prompt = f"""
你是***，从下面的文本里抽取结构化信息，你的回复必须是一个 JSON 对象，不要有任何其他文字。

字段：
- name
- age
- sex
- ...

信息
{TEXT}
"""
```

## 调用API传入 response_format: {"type": "json_object"}

```python
from openai import OpenAI

client = OpenAI(api_key="sk-****", base_url="https://api.deepseek.com")
resp = client.chat.completions.create(
    model="deepseek-flash",
    messages=[{"role": "user", "content": '请输出 JSON 对象，例如 {"name":"张三"}。文本：张三'}],
    temperature=0,
    response_format={"type": "json_object"},
)
```
注意事项（deepseek文档）：
- 用户传入的 system 或 user prompt 中必须含有 json 字样，并给出希望模型输出的 JSON 格式的样例，以指导模型来输出合法 JSON。
- 需要合理设置 max_tokens 参数，防止 JSON 字符串被中途截断。  
- 在使用 JSON Output 功能时，API 有概率会返回空的 content。我们正在积极优化该问题，您可以尝试修改 prompt 以缓解此类问题。
