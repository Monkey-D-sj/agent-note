"""
用 OpenAI 的 SDK 调 DeepSeek，测模型 JSON 输出的三件事

1. 语法 —— 是不是合法 JSON
2. 结构 —— 符不符合 schema（同时对比 prompt_only / json_object 两种约束方式）
3. 内容 —— 抽出来的值对不对（跟 tasks.json 里的 expected 比）

第 3 项才是「抽取质量」本身。前两项全过但内容全错是完全可能的
（比如 {"title": "随便写的"} 格式一点问题没有），所以这三个数字
必须分开看，不能揉成一个「准确率」。

顺带：json_object 模式有概率返回空 content，这是官方承认的已知问题。

任务定义在 tasks.json 里，加新任务往里加一条就行，不用动这个文件。

用法（在项目根目录执行）：
    uv run json_output/json_output.py                     # 非思考模式，每组跑 5 轮
    uv run json_output/json_output.py 1 --compare-thinking # 四组对照，每组跑 1 轮
    uv run json_output/json_output.py 10 --compare-thinking
"""

import argparse
import json
import os
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from jsonschema import ValidationError, validate
from openai import OpenAI

# Windows 控制台默认是 GBK，直接 print 中文 / √ × 会乱码，强制切 UTF-8。
# stderr 也要切：SystemExit 带字符串时是打到 stderr 的。
for _stream in (sys.stdout, sys.stderr):
	if hasattr(_stream, "reconfigure"):
		_stream.reconfigure(encoding="utf-8")

load_dotenv()

# ---------------------------------------------------------------- 模型接入
# 关键点：DeepSeek 的接口是 OpenAI 兼容的，所以直接用 openai 这个 SDK 就行，
# 只是把 base_url 和 model 换掉。哪天要换成别家，改的也是这两行。
MODEL = "deepseek-flash"
TASKS_PATH = Path(__file__).with_name("tasks.json")

# 显式给输出上限。推理模型（deepseek-flash）的思考 token 也从这里扣，
# 不设的话任务一长就可能被静默截断——content 变成空串，而 finish_reason
# 只报 length，接口不报错，看起来像是模型「什么都没说」。
# 只按实际用量计费，所以设宽不花钱。
MAX_TOKENS = 4096


def make_client():
	api_key = os.getenv("DEEPSEEK_API_KEY")
	if not api_key:
		raise SystemExit("缺少 DEEPSEEK_API_KEY。请先 `Copy-Item .env.example .env` 并填入 key。")
	base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
	return OpenAI(api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------- 任务加载
def load_tasks(path=TASKS_PATH):
	"""从 tasks.json 读任务列表"""
	if not Path(path).exists():
		raise SystemExit(f"找不到任务文件 {path}")
	tasks = json.loads(Path(path).read_text(encoding="utf-8")).get("tasks", [])
	if not tasks:
		raise SystemExit(f"{path} 里没有任务")
	return tasks


def build_prompt(task, variant=None):
	"""填 prompt 里的占位符。

	{text}          -> 任务自己的 text
	variant 里的键  -> 对应值（比如 {missing_rule}）

	用 replace 而不是 str.format：prompt 里多半会写 JSON 示例，
	那些 { } 会被 format 当成占位符直接报错。
	"""
	values = {"text": task["text"], **(variant or {})}
	prompt = task["prompt"]
	for key, val in values.items():
		if key != "name":
			prompt = prompt.replace("{" + key + "}", val)
	return prompt


# ---------------------------------------------------------------- 调用方式
def call_model(client, prompt, json_mode, thinking):
	"""JSON 约束和思考模式是两个独立的实验维度。只取最终 content 判分。"""
	options = {
		"model": MODEL,
		"messages": [{"role": "user", "content": prompt}],
		"max_tokens": MAX_TOKENS,
		"extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}},
	}
	if json_mode:
		options["response_format"] = {"type": "json_object"}
	# DeepSeek 文档说明思考模式会接受但忽略 temperature；重复实验仍需统计波动。
	options["temperature"] = 0
	resp = client.chat.completions.create(**options)
	choice = resp.choices[0]
	reasoning = getattr(choice.message, "reasoning_content", None)
	return choice.message.content, choice.finish_reason, len(reasoning or "")


def strategies(compare_thinking):
	modes = [False, True] if compare_thinking else [False]
	for thinking in modes:
		for json_mode in (False, True):
			name = ("json_object" if json_mode else "prompt_only") + ("+thinking" if thinking else "+no_thinking")
			yield name, json_mode, thinking


# ---------------------------------------------------------------- 结果判定
@dataclass
class Result:
	ok_json: bool              # 能不能解析成 JSON
	ok_schema: bool            # 是否符合该任务的 schema
	note: str                  # 失败原因 / 备注
	data: dict | None = None   # 解析出来的对象，留着给内容比对用


def strip_fence(text):
	"""剥掉 markdown 代码块包裹 —— prompt-only 最常见的翻车形式之一"""
	t = text.strip()
	if not t.startswith("```"):
		return t
	# 去掉开头那行 ``` 或 ```json
	_, _, rest = t.partition("\n")
	if rest:
		t = rest
	if t.rstrip().endswith("```"):
		t = t.rstrip()[:-3]
	return t.strip()


def check(raw, schema):
	"""只判语法和结构两层。内容对不对是另一回事，见 compare()"""
	if not raw or not raw.strip():
		return Result(False, False, "返回空内容")

	try:
		data = json.loads(raw)
	except json.JSONDecodeError as e:
		# JSON 输出基准必须按原始 content 判分；修复后的可解析性只作诊断。
		try:
			json.loads(strip_fence(raw))
		except json.JSONDecodeError:
			return Result(False, False, f"不是合法 JSON（{e.msg} @ 位置 {e.pos}）")
		return Result(False, False, "原始输出带代码围栏，剥掉后才可解析")

	try:
		validate(instance=data, schema=schema)
	except ValidationError as e:
		loc = "/".join(str(p) for p in e.absolute_path) or "(根)"
		return Result(True, False, f"JSON 合法但不符合 schema：{loc} -> {e.message}", data)

	return Result(True, True, "OK", data)


# ---------------------------------------------------------------- 内容比对
# 前三层里最容易糊弄的一层：格式全对、内容全错，一样能拿满分。
# 所以单独把「跟标准答案比」拎出来。
_MISSING = object()


def _norm(v):
	"""字符串去掉首尾空白。只影响比较，不动原值。"""
	return v.strip() if isinstance(v, str) else v


def _canon(v):
	"""把值转成一个稳定的比较键。

	关键在 sort_keys：dict 的键顺序不该影响结果。
	这里原来用的是 str()，而 str({"a": 1, "b": 2}) != str({"b": 2, "a": 1})——
	内容相同的两个对象数组会排到不同位置，被判成不一样（假失败，
	模型没错但记它一penalty）。
	"""
	return json.dumps(_norm(v), sort_keys=True, ensure_ascii=False)


def _field_ok(act, exp, mode):
	"""单个字段算不算对。act 实际值，exp 期望值"""
	if act is _MISSING:
		return False
	if isinstance(exp, list):
		if not isinstance(act, list):
			return False
		# 数组一律顺序无关：["a","b"] 和 ["b","a"] 算一样；
		# 元素是对象时（items / decisions 这类），键顺序也不该影响
		a = [_canon(x) for x in act]
		e = [_canon(x) for x in exp]
		if mode == "contains":
			return all(x in a for x in e)
		return sorted(a) == sorted(e)
	if mode == "contains":
		return isinstance(act, str) and isinstance(exp, str) and exp in act
	return _norm(act) == _norm(exp)


def compare(actual, expected, modes=None):
	"""拿模型输出跟标准答案逐字段比。

	modes 可以给单个字段换比较方式：
	  exact（默认）—— 完全相等
	  contains    —— 实际值包含期望值即可
	                 （用在「至少得有这些」这种本来就宽松的要求上，
	                  否则标准答案会把模型的合理发挥错判成错误）

	返回 (是否全对, 差异明细, 对的字段数, 总字段数)
	"""
	modes = modes or {}
	if not isinstance(actual, dict):
		return False, [f"顶层不是对象（{type(actual).__name__}），没法逐字段比"], 0, len(expected)

	diffs = []
	ok_n = 0
	for key, exp in expected.items():
		act = actual.get(key, _MISSING)
		if _field_ok(act, exp, modes.get(key, "exact")):
			ok_n += 1
		else:
			shown = "<缺失>" if act is _MISSING else repr(act)
			diffs.append(f"{key}: 期望 {exp!r}，实际 {shown}")
	return ok_n == len(expected), diffs, ok_n, len(expected)


# ---------------------------------------------------------------- 输出排版
def _dw(s):
	"""终端显示宽度：中文/全角算 2，其余算 1。不这么算表格会歪。"""
	return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def _pad(s, width):
	return s + " " * max(0, width - _dw(s))


# ---------------------------------------------------------------- 跑测试
def run(rounds=5, compare_thinking=False):
	client = make_client()
	tasks = load_tasks()
	print(f"model={MODEL}   任务 {len(tasks)} 个   每组 {rounds} 轮   对照思考模式={compare_thinking}")
	print("符号：√ 内容全对   × 内容有错   J 语法/结构没过   E 调用报错")

	table = []

	for task in tasks:
		tname = task["name"]
		schema = task["schema"]
		expected = task.get("expected")
		if not expected:
			raise SystemExit(f"任务 {tname} 没有 expected（标准答案），没法做内容比对。")
		modes = task.get("compare") or {}
		# 没写 variants 的任务就当成只有一个「无名变体」，跑一遍
		variants = task.get("variants") or [None]

		for variant in variants:
			vname = (variant or {}).get("name", "")
			label = f"{tname} / {vname}" if vname else tname
			prompt = build_prompt(task, variant)

			print(f"\n########## {label} —— {task.get('description', '')}")
			prev_sig = None   # 上一组的错法，用来省掉重复打印

			for sname, json_mode, thinking in strategies(compare_thinking):
				marks = []         # 每轮一个符号
				errs = {}          # 错误描述 -> 出现次数（不含原始输出，否则同一种错会因为
				                   # 原始输出的细微差异被拆成多条，计数就废了）
				samples = {}       # 错误描述 -> 第一次见到的原始输出，只当例子展示
				odd_finish = {}    # 非 stop 的 finish_reason
				n_json = n_schema = n_content = 0
				f_ok = f_total = 0

				for _ in range(rounds):
					try:
						raw, finish_reason, reasoning_chars = call_model(
							client, prompt, json_mode, thinking)
					except Exception as e:
						marks.append("E")
						k = f"调用报错 {type(e).__name__}: {str(e)[:90]}"
						errs[k] = errs.get(k, 0) + 1
						continue

					if finish_reason not in ("stop", None):
						odd_finish[finish_reason] = odd_finish.get(finish_reason, 0) + 1

					r = check(raw, schema)
					n_json += r.ok_json
					n_schema += r.ok_schema

					# 语法或结构没过，就没法谈内容了
					if not (r.ok_json and r.ok_schema):
						marks.append("J")
						errs[r.note] = errs.get(r.note, 0) + 1
						samples.setdefault(r.note, (raw or "").strip().replace("\n", "\\n")[:40])
						continue

					# 三层全过才算真做对
					all_ok, diffs, fk, ft = compare(r.data, expected, modes)
					n_content += all_ok
					f_ok += fk
					f_total += ft
					marks.append("√" if all_ok else "×")
					for d in diffs:
						errs[d] = errs.get(d, 0) + 1

				strip = "".join(marks)
				if len(strip) > 60:
					strip = strip[:60] + "…"
				rate = f"{f_ok}/{f_total} ({f_ok / f_total:.0%})" if f_total else "—"
				print(f"  {_pad(sname, 26)}{_pad(strip, 24)}"
				      f"内容 {n_content:>3}/{rounds}   字段 {rate}")

				for fr, cnt in odd_finish.items():
					print(f"  {'':<26}⚠ finish_reason={fr} ×{cnt}（可能被截断）")

				if errs:
					sig = tuple(sorted(errs.items()))
					if sig == prev_sig:
						print(f"  {'':<26}└─ 错法同上一组，不重复列")
					else:
						for msg, cnt in sorted(errs.items(), key=lambda kv: (-kv[1], kv[0])):
							ex = f"   例: {samples[msg]!r}" if msg in samples else ""
							print(f"  {'':<26}✗ {msg}   ×{cnt}{ex}")
					prev_sig = sig
				else:
					prev_sig = None

				table.append((label, sname, n_json, n_schema, n_content, f_ok, f_total))

	# 汇总表：每一列都按各自最大宽度补齐（最后一列也补），
	# 这样所有行的显示宽度完全一致，不会看着歪。
	print("\n" + "=" * 76)
	print("汇总")
	rows = [["任务 / 策略", "语法", "结构", "内容全对", "字段准确率"]]
	for label, sname, n_json, n_schema, n_content, f_ok, f_total in table:
		rate = f"{f_ok}/{f_total} ({f_ok / f_total:.0%})" if f_total else "—"
		rows.append([f"{label} / {sname}", f"{n_json}/{rounds}", f"{n_schema}/{rounds}",
		             f"{n_content}/{rounds}", rate])
	widths = [max(_dw(r[i]) for r in rows) + 2 for i in range(len(rows[0]))]
	for r in rows:
		print("  " + "".join(_pad(c, w) for c, w in zip(r, widths)))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="对比 DeepSeek 的 JSON 输出约束与思考模式")
	parser.add_argument("rounds", nargs="?", type=int, default=5, help="每个任务每组的运行轮数")
	parser.add_argument("--compare-thinking", action="store_true", help="同时测试思考模式开启和关闭")
	args = parser.parse_args()
	if args.rounds < 1:
		parser.error("rounds 必须大于 0")
	run(args.rounds, args.compare_thinking)
