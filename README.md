# python线程

## 协程
asyncio
单线程上执行，要使用**async await**的写法
```python
import asyncio
async def io_func(name):
	# do something
	return name
def sync_func():
	# do something
	return ""

async def test():
	await io_func(12345) # ✅ 异步函数
	await asyncio.to_thread(sync_func)   # ✅ 同步函数用 to_thread 包
	results = await asyncio.gather(
		io_func("a"),
		io_func("b"),
		io_func("c"),
	) # 结果按传入顺序返回，但完成顺序不保证
	return results

async def test2():
	# 裸协程：不跑，等 as_completed 内部帮你包
	tasks = (
		io_func("a"), # 不会像同步直接执行，返回一个协程对象，不会执行，类似于Promise对象
		io_func("b"),
		io_func("c"),
	)
	# 或者自己包成 Task：立即开始跑
	tasks = [
		asyncio.create_task(io_func("a")), # 会开始执行
		asyncio.create_task(io_func("b"))
	]
	results = []
	for result in asyncio.as_completed(tasks): # 谁先返回谁进
		results.append(await result)
	return results

print(asyncio.run(test()))
print(asyncio.run(test2()))
```
事件循环，IO会让出，让后面接着执行
CPU密集计算会卡住

## 线程
thread

普通写法

```python
import threading

def test(name):
    print(f"{name} do something")

t1 = threading.Thread(target=test, args=("python"))
t2 = threading.Thread(target=test, args=("c++"))
# 开始
t1.start()
t2.start()
# 在当前线程执行
# t1.run()
# t2.run()
# 等待执行完成
t1.join()
t2.join()
```
这样写拿不到**test**返回的结果

更常用写法
```python
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def call_llm(name, seconds):
	print(f"{name} 开始调用")
	time.sleep(seconds)
	print(f"{name} 调用完成")
	return f"{name}!"

with ThreadPoolExecutor(max_workers=3) as executor:
	futures = {
		executor.submit(call_llm, "deepseek", 2): "deepseek",
		executor.submit(call_llm, "openai", 5): "openai",
		executor.submit(call_llm, "claude", 1): "claude",
		executor.submit(call_llm, "claude2", 2): "claude2",
	}
	# timeout可选
	for future in as_completed(futures, timeout=5):
		name = futures[future]
		result = future.result()
		print(f"{name} {result}")
```
可以通过result拿到结果，as_completed会让先执行完的返回，否则会按照顺序

### **GIL（Global Interpreter Lock/全局解释器锁）**
仅限于CPython，当然也是绝大多数了
因为GIL的存在，多线程同时只能有一个线程执行python代码
#### 什么时候释放
- IO，主动让出
- 5ms，强制让出（所以耗时操作不会卡住GIL导致其他线程无法进行）
- 其他语言拓展，如c、c++等
- 后果：
  - IO 密集：多线程有效
  - CPU 密集：多线程≈单线程，用多进程

## 线程
用法和thread类似，多了terminate，可以杀掉进程

## 选型
| 场景 | 用什么 |
|---|---|
| 大量 IO、库是异步的 | asyncio |
| 大量 IO、库是同步的 | ThreadPoolExecutor / asyncio.to_thread |
| CPU 密集 | ProcessPoolExecutor |
| 简单后台线程 | threading.Thread |

## 协程 vs 线程
- 协程：单线程内**协作式**调度，切换点只在 await，开销小，能开成千上万
- 线程：**抢占式**调度，切换不可控，每个线程有栈开销，开几百个到顶
- 协程不怕数据竞争，但注意 await 之间状态可能被改