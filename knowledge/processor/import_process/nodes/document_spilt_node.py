import os
import json
import re
from sqlite3.dbapi2 import paramstyle
from typing import Tuple, List, Dict, Any

from knowledge.processor.import_process.base import BaseNode,setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.md_utils import MarkdownTableLinearizer

from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentSplitNode(BaseNode):

    name = "DocumentSplitNode"

    def process(self, state:ImportGraphState)->ImportGraphState:
        # 为什么要切割？1. 嵌入模型语义更准确。2.注入元数据。3.多路召回。4.性能、成本高--》减少LLM的幻觉，提高检索质量

        # 1. 获取参数
        md_content, file_title, max_content_length, min_content_length= self._get_inputs(state)
        # 2. 根据标题切割
        sections = self._spilt_by_headings(md_content, file_title)

        # 3.处理
        #  section内容过长，进行二次切割,  section内容过短，看能否合并。同属与一个标题下的才能进行处理
        final_chunks = self._split_and_merge(sections, max_content_length,min_content_length)
        # 4. 组装
        chunks = self._assemble_content(final_chunks)

        # Step 6: 日志统计
        self._log_summary(md_content, chunks, max_content_length)

        # Step 7: 备份
        state["chunks"] = chunks
        self._backup_chunks(state, chunks)
        return state

    def _log_summary(self, raw_content: str, chunks: List[dict], max_length: int):
        """输出切分统计信息"""
        self.log_step("step5", "输出统计")

        lines_count = raw_content.count("\n") + 1
        self.logger.info(f"原文档行数: {lines_count}")
        self.logger.info(f"最终切分章节数: {len(chunks)}")
        self.logger.info(f"最大切片长度: {max_length}")

        if chunks:
            self.logger.info("章节预览:")
            for i, sec in enumerate(chunks[:5]):
                title = sec.get("title", "")[:30]
                self.logger.info(f"  {i + 1}. {title}...")
            if len(chunks) > 5:
                self.logger.info(f"  ... 还有 {len(chunks) - 5} 个章节")

    def _backup_chunks(self, state: ImportGraphState, sections: List[dict]):
        """将切分结果备份到 JSON 文件"""
        self.log_step("step6", "备份切片")

        local_dir = state.get("file_dir", "D:\A-py\AI_project\smartku\knowledge\processor\import_process\import_temp_dir")
        if not local_dir:
            self.logger.debug("未设置 file_dir，跳过备份")
            return

        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            self.logger.info(f"已备份到: {output_path}")

        except Exception as e:
            self.logger.warning(f"备份失败: {e}")



    def _get_inputs(self, state: ImportGraphState) -> Tuple[str, str, int,int]:

        self.log_step("step1", "切分文档的参数校验以及获取...")

        config = get_config()
        # 1. 获取md_content
        md_content = state.get('md_content')

        # 2. 统一换行符
        if md_content:
            md_content = md_content.replace("\r\n","\n").replace("\r","\n")

        # 3.获取文件标题
        file_title = state.get('file_title')

        # 4. 校验最大最小值
        if config.max_content_length <=0 or config.min_content_length <=0 or config.max_content_length <= config.min_content_length:
            raise ValueError(f"切片长度参数校验失败")

        return md_content, file_title, config.max_content_length, config.min_content_length



    def _spilt_by_headings(self, md_content: str, file_title: str)  -> Tuple[List[dict]]:

        """
        根据MD的标题（1-6）级标题进行
        flush核心任务是：
        当遇到新标题（或文档结束）时，把之上一个标题及其累积的正文，打包成一个完整的对象，并计算出它的“父标题”，最后存入结果列表
        首先将之前累积的内容合并一起，
        然后定义一个hierarchy全局数组来实时记录当前所在的标题路径，
        并从当前级别的上一级开始倒序查找，找到最近的非空标题作为 parent_title，若找不到上级：则父标题设为自身（根节点）或文件名
        同时
        Args:
            md_content: md内容
            file_title: 文档名字

        Returns:
            Tuple[List[dict],bool]
            List[dict]:Sections
            bool: md文档是否有标题

            {
            “title": "# 第一章"  -》当前段的标题
            "body": " 正文内容"
            "file_title" : "万用表"
            "parent_title": " # 第一章" 父标题会更新
            }
        """
        self.log_step("step2", "根据标题进行切分...")

        #1. 定义变量
        in_fence = False
        body_lines = []
        current_title = ""
        current_level = 0
        hierarchy = [""] * 7   #(第一个位置不用)
        sections = []
        # 2. 定义正则表达式
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")

        # 3. 切分
        content_lines = md_content.split("\n")

        def _flush():
            """
            封装section对象
            Returns:

            """
            body = "\n".join(body_lines)

            if current_title or body:
                parent_title = ""
                # 向前遍历数组，找到的第一个非空的标题就是父标题
                for i in range(current_level-1 ,0 , -1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break

                if not parent_title:
                    #兜底策略
                    #如果上面的循环跑完了也没找到父标题（说明当前标题已经是顶级，比如 H1，或者文档开头没有标题的部分）：
                    parent_title = current_title if current_title else file_title

                return sections.append({
                    "title" : current_title if current_title else file_title,
                    "body": body,
                    "file_title": file_title,
                    "parent_title": parent_title,
                })


        for content_line in content_lines:
            # 3.1 判断当前是否存在代码块围栏
            if content_line.strip().startswith("```") or content_line.strip().startswith("~~~"):
                in_fence = not in_fence
            match = heading_re.match(content_line) if not in_fence else None

            if match:
                #如果当前行是标题
                _flush()
                level = len(match.group(1)) # 拿当前标题的级别
                current_level = level
                current_title = content_line
                hierarchy[level] = current_title
                # 存储当前遍历的标题
                body_lines = []

                for i in range(level+1 , 7):  # 清空
                    hierarchy[i] = ""
            # 把除了标题之外的全部内容收集起来
            else:
                body_lines.append(content_line)
        _flush()

        return sections

    def _split_and_merge(self, sections: List[Dict[str,Any] ], max_content_length: int, min_content_length: int):
        """
        合并时，只对同一个Parent以及body小于最小值的块进行操作
        Args:
            sections:
            max_content_length: 每一个section的content。title+body
            min_content_length: 如果比这个小就要进行合并

        Returns:

        """
        current_sections = []
        # 1.qiefen
        for section in sections:
           current_sections.extend(self.split_long_section(section, max_content_length))


        # 2. 合并
        final_sections = self.merge_short_section(current_sections, min_content_length)

        # 3. 返回
        return final_sections

    def split_long_section(self, section:List[Dict[str,Any]], max_content_length:str):
        """
        只有达到了这个max_content_length才会进行切分
        Args:
            section:
            max_content_length:

        Returns:

        """
        self.log_step("step3","进行长内容的切分")

        # 1. 获取section对象属性
        title = section.get('title')
        body = section.get('body')
        file_title = section.get('file_title')
        parent_title = section.get('parent_title')

        #2. 判断表格
        if "<table>" in body:
            self.logger.info("检测到了表格")
            body = MarkdownTableLinearizer.process(body)


        # 2.单独对标题进行校验
        title_max_length = 50
        if len(title) > title_max_length:
            self.logger.warning(f"当前文件{file_title}对应的{title}过长")
            title = title[:50]

        # 3.拼接title前缀
        title_prefix = f"{title}\n\n"

        # 4 计算总长度
        total_length = len(title_prefix) + len(body)

        # 5. 判断小于或者刚好满足阈值
        if total_length <= max_content_length:
            return [section]

        # 6. 计算body可用的长度
        body_length = max_content_length - len(title_prefix)
        if body_length <= 0:
            return [section]
        # 7. 切分
        # 7.1 定义切分器
        text_spilter = RecursiveCharacterTextSplitter(
            chunk_size=body_length,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " ",""],
            keep_separator=False
        )

        # 7.2 进行切割
        texts = text_spilter.split_text(body)

        # 判断(body没有，或者标题下面的内容很少
        if len(texts) <= 1:
            return [section]

        sub_sections = []
        for index,text in enumerate(texts):
            sub_sections.append({
                "title": title+'-'+f"{index+1}",
                "body": text,
                "file_title": file_title,
                "parent_title": parent_title,
                "part": f"{index+1}"
            })
        return sub_sections

    def merge_short_section(self, current_sections:List[Dict[str,Any]], min_content_length:str):
        """
        贪心累加算法
        局限：1、最后合的块会超过最小的阈值（其实不会特别大）
        if same_parent and len(current_section.get('body')) < min_content_length
        例如此时current_section_body=450，下一个块是section块是800，而最小的阈值是500，依旧会把下一个section加进去
            2、最后留下的单独小块（有一点点无所谓）
        Args:
            current_sections:
            min_content_length:

        Returns:

        """
        # 把section1拿出来
        current_section = current_sections[0]
        final_sections = [] # 合并最终的

        # 从section2开始遍历
        for next_section in current_sections[1:]:

            same_parent = (current_section['parent_title'] == next_section['parent_title'])
            # 如果此时section2的长度小于min，且同属于parent
            if same_parent and len(current_section.get('body')) < min_content_length:
                # 1 合并body
                current_section['body'] = (
                    current_section.get('body').rstrip() +"\n\n"+  next_section.get('body').lstrip()
                )

                # 还要更新current_title
                current_section['title'] = current_section['parent_title']

                # 只要满足条件进到if语句
                # 比如此时section0是原来的（无part），s1是切分后的；或者s0及后续的都是原来的，都直接将此时section中part属性赋值(只要有值就行，
                current_section['part'] = 0

            else:
                # 1. 将原来的current_section保存在一块
                final_sections.append(current_section)

                # 2. 更新next_section
                current_section = next_section

        # 最后结束for循环curren_section还会有值，不要忘了封装
        final_sections.append(current_section)

        #对part做处理（给每个父标题设置一个Part计数器）
        part_cnt = {}
        res = []
        for final_section in final_sections:
            # 没有被切分器进行切分的section没有part，所以要判断
            if 'part' in final_section:
                # 获取此时section中父节点的值
                parent_title = final_section.get("parent_title")
                # 给计数器赋值
                part_cnt[parent_title] = part_cnt.get(parent_title,0)+1
                # 获取计数器的值
                new_part = part_cnt[parent_title]

                final_section['part'] = new_part

                final_section['title'] = final_section['title'] + f"- {new_part}"
            res.append(final_section)
        return res

    def _assemble_content(self, final_chunks:List[Dict[str,Any]])->List[Dict[str,Any]]:
        """
        Args:
            final_chunks:

        Returns:

        """
        chunks = []
        #1. 获取chunk的信息
        for chunk in final_chunks:
            title = chunk.get("title")
            file_title = chunk.get("file_title")
            parent_title = chunk.get("parent_title")
            body = chunk.get("body")
            content = f"{title}\n\n{body}"
            #2. 构建最终对象
            assemble_chunk={
                "title": title,
                "file_title": file_title,
                "content": content,
                "parent_title": parent_title,
            }

            # 3. 判断part是否存在
            if "part" in chunk:
                assemble_chunk['part'] = chunk.get('part')

            chunks.append(assemble_chunk)
        return chunks

if __name__ == '__main__':
    setup_logging()

    document_node = DocumentSplitNode()

    file_path = r"D:\A-py\AI_project\smartku\knowledge\processor\import_process\import_temp_dir\万用表RS-12的使用\hybrid_auto\万用表RS-12的使用.md"
    with open (file_path, "r" , encoding="utf-8") as f:
         content = f.read()
    state = {
        "file_title": "万用表的使用",
        "md_content": content,
    }
    document_node.process(state)









