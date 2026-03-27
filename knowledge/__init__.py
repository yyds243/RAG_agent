import asyncio
import time  # 导入 time 模块

async def task1():
    start_time = time.time()  # 记录任务1开始的时间
    print("Task 1 started")
    await asyncio.sleep(2)  # 模拟 I/O 操作，任务暂停2秒
    end_time = time.time()  # 记录任务1结束的时间
    print("Task 1 completed")
    print(f"Task 1 took {end_time - start_time:.2f} seconds")

async def task2():
    start_time = time.time()  # 记录任务2开始的时间
    print("Task 2 started")
    await asyncio.sleep(1)  # 模拟 I/O 操作，任务暂停1秒
    end_time = time.time()  # 记录任务2结束的时间
    print("Task 2 completed")
    print(f"Task 2 took {end_time - start_time:.2f} seconds")

async def main():
    start_time = time.time()  # 记录整个任务开始的时间
    # 启动两个任务，任务会并行进行
    await asyncio.gather(task1(), task2())
    end_time = time.time()  # 记录整个任务完成的时间
    print(f"Total time: {end_time - start_time:.2f} seconds")

# 运行异步任务
asyncio.run(main())