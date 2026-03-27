"""
导入流程节点基类

为工作流中的每一个节点提供一套标准的接口、日志记录以及异常处理的机制。

定义统一的节点接口规范，提供通用功能
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Optional
import logging

from knowledge.processor.import_process.config import ImportConfig, get_config
from knowledge.processor.import_process.exceptions import ImportProcessError
T = TypeVar("T")  # 泛型状态类型


class BaseNode(ABC):
    # 当实例某一个子节点时候，
    """
    导入流程节点基类

    所有节点类都应继承此基类，实现 process 方法。
    基类提供统一的日志、任务追踪和错误处理。

    使用示例:
        class MyNode(BaseNode):
            name = "my_node"

            def process(self, state):
                # 实现具体逻辑
                return state

        # 作为 LangGraph 节点使用
        node = MyNode()
        workflow.add_node("my_node", node)
    """

    name: str = "base_node"  # 节点名称，子类应覆盖

    def __init__(self, config: Optional[ImportConfig] = None):
        """
        初始化节点

        Args:
            config: 配置对象，默认使用全局配置
        """
        self.config = config or get_config()
        self.logger = logging.getLogger(f"import.{self.name}")  #通过日志器来对每个节点初始化一个日志对象

    def __call__(self, state: T) -> T:
        """
        节点执行入口
        当把实例当成方法来使用会自动调用call方法
        LangGraph 调用节点时会调用此方法。
        提供统一的日志输出、任务追踪和异常处理。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典

        Raises:
            ImportProcessError: 节点执行失败时抛出
        """

        self.logger.info(f"--- {self.name} 开始 ---")


        try:
            result = self.process(state)  # 调用子节点的处理逻辑
            self.logger.info(f"--- {self.name} 完成 ---")
            return result
        except ImportProcessError:
            # 已经是自定义异常，直接抛出
            raise
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise ImportProcessError(
                message=str(e),   # 异常内容描述
                node_name=self.name,   # 哪一个节点
                cause=e  # 异常原因
            )

    @abstractmethod
    def process(self, state: T) -> T:
        """
        节点核心处理逻辑

        子类必须实现此方法。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典
        """
        pass

    def log_step(self, step_name: str, message: str = ""):
        #__call__ 方法只记录了节点的“开始”和“完成”。但在一个复杂的节点内部，我们需要知道内部处理的流程。
        #log_step 允许开发者在 process 方法内部，通过简单的一行代码记录关键进度，而不需要每次都去处理复杂的日志格式
        """
        记录步骤日志

        Args:
            step_name: 节点的某一步的步骤
            message: 附加信息
        """
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)


# 配置日志格式
def setup_logging(level: int = logging.INFO):
    """
    配置导入流程日志

    Args:
        level: 日志级别
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
