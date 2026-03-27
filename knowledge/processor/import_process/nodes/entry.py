import json
from pathlib import Path

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.exceptions import ValidationError

class EntryNode(BaseNode):
    """
    实体节点，第一个节点
    作用：对上传的文件类型进行判断
    """
    name = "entry"

    def process(self,state: ImportGraphState) -> ImportGraphState:
        """
        处理文件类型的逻辑
        Args:
            state: ImportGraphState 该节点处理之前的节点状态

        Returns: ImportGraphState 处理之后的节点状态

        """
        # 1. 获取导入的文件路径以及所在的目录
        self.log_step("step1","获取文件路径")
        import_file_path = state.get("import_file_path")
        file_dir  = state.get("file_dir")

        # 2. 简单校验文件路径以及目录
        self.log_step("step2","检测文件路径")
        if not file_dir or not import_file_path:
            raise ValidationError("文件目录或路径不存在",self.name)

        # 3.path操作
        path = Path(import_file_path)

        # 4. 获取上传文件的后缀
        suffix = path.suffix.lower()

        # 5.判断文件后缀
        if suffix == '.pdf':
            state['is_pdf_read_enabled'] = True
            state['pdf_path'] = import_file_path

        elif suffix == '.md':
            state['is_md_read_enabled'] = True
            state['md_path'] = import_file_path
        else:
            self.logger.debug(f"文件类型{suffix}不支持")
            raise ValidationError(f"文件类型{suffix}不支持")

        # 6.获取文件的标题名
        file_title = path.stem
        state['file_title'] = file_title

        return state

if __name__ == '__main__':
    # pdf_path = Path(r"D:\A-py\AI_project\smartku\knowledge\processor\import_process\import_temp_dir\万用表RS-12的使用.pdf")
    # print(pdf_path.name)
    # print(pdf_path.suffix)
    # print(pdf_path.stem)

    #方法一： 直接实例该节点对象，调用process方法
    #方法二： 把实例当方法使用
    setup_logging()
    # 1. 构建该节点需要的state
    test_entry_state = {
        "file_dir": r"D:\A-py\AI_project\smartku\knowledge\processor\import_process\import_temp_dir",
        "import_file_path": r"D:\A-py\AI_project\smartku\knowledge\processor\import_process\import_temp_dir\万用表RS-12的使用.pdf"
    }

    # 2.实例EntryNode节点
    entry_node = EntryNode()

    # 3.调用process方法
    process_state = entry_node.process(test_entry_state)

    print(json.dumps(process_state,indent=3))




