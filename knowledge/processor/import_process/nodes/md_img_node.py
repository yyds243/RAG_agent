import logging
import os, re
import re
import time
import base64
from collections import deque
from pathlib import Path
from typing import Tuple, List, Deque

from openai import OpenAI

from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import ValidationError, ImageProcessingError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.minio_utils import get_minio_client


class MarkDownImageNode(BaseNode):
    """
    处理md图片节点类
    """
    name = "MarkDownImageNode"
    def process(self, state: ImportGraphState) -> ImportGraphState:
        """

        Args:
            state:上一个节点处理之后的state最新状态

        Returns: 当前节点处理之后的state最新状态

        """
        config = get_config()

        # 1.处理文件路径（要返回md内容，md地址，图片目录这三个参数）
        md_content, md_path_obj, image_dir = self._get_img_md_content(state)
        if not image_dir.exists():
            # 就不需要处理图片了
            self.logger.info(f"文件{md_path_obj}PDF中没有图片要处理")
            state['md_content'] = md_content
            return state
            
        # 2.扫描并处理图片
        target_images_context = self._scan_images_and_context( image_dir, md_content, config)

        # 3. 用VLM给图片生成图片描述(摘要)
        images_summaries = self._extract_img_summary(md_path_obj.stem, target_images_context,config)


        # 4.将本地图片上传到minio--》remote_url(图片远程的地址）
        # 4.1 替换md图片中的本地url，还要替换vlm生成的摘要
        new_md_content = self._upload_img_and_update_new_md(md_path_obj.stem, md_content,images_summaries,target_images_context,config)

        # 5.备份更新后的内容
        self._backup_new_md_file(md_path_obj, new_md_content)

        # 6. 更新State
        state['md_content'] = new_md_content
        # 7. 返回state
        return state

    def _backup_new_md_file(
            self,
            md_path_obj: Path,
            new_md_content: str
    ) -> str:
        self.log_step("step_5", "备份新文件")

        new_file_path = md_path_obj.with_name(
            f"{md_path_obj.stem}_new{md_path_obj.suffix}"
        )

        try:
            with open(new_file_path, "w", encoding="utf-8") as f:
                f.write(new_md_content)
            self.logger.info(f"处理后的文件已备份至: {new_file_path}")
        except IOError as e:
            self.logger.error(f"写入新文件失败 {new_file_path}: {e}")
            raise ImageProcessingError(f"文件写入失败: {e}", node_name=self.name)
        return str(new_file_path)


    def _get_img_md_content(self, state:ImportGraphState) -> Tuple[str, Path, Path]:
        """

        Args:
            state:  上一个节点处理之后state的最新状态

        Returns:
            md_content
            md_path_obj
            image_dir
        """

        self.log_step("step1","读取md内容以及图片目录")
        # 1.获取md_path
        md_path = state.get('md_path','')

        # 2.判断路径是否有内容
        if not md_path:
            raise ValidationError("md文件不存在",self.name)

        # 3.标准化
        md_path_obj = Path(md_path)
        # 4. 判断路径是否有效
        if not md_path_obj.exists():
            raise ValidationError("md路径无效",self.name)

        # 5.读取md内容
        with open(md_path_obj,'r',encoding='utf-8')as f:
            md_content = f.read()

        # 6.构建图片目录
        image_dir = md_path_obj.parent / 'images'

        # 7.fanhui
        return md_content, md_path_obj, image_dir

    def _scan_images_and_context(self, image_dir:Path, md_content:str,config)->List[Tuple[str,str,Tuple[str,str,str]]]:
        """
        扫描并处理图片
        返回所有有效图片的丰富信息（image_name,image_path,图片的上下文）
        图片上下文策略： 1.先找到当前图片的最近一个标题（定位到标题的位置以及标题的内容）
                      2. 从图片上面的内容开始向上找，直到找到离图片最近的标题下面为止、、
                      3. 根据开始索引和结束索引定位到两个索引间的内容、
                      4. 利用段落和最大字符数选择从这个区域中最终留下多少
        Args:
            image_dir: 图片目录
            md_content: md内容
            config: 配置信息

        Returns:List[Tuple[str,str,Tuple[str,str,str]]]
        List[("图片名字“，"图片地址"，（"离图片最近的上面一个标题","图片的上文","图片的下文"))]

        """
        self.log_step("step2",f"扫描图片文件目录{image_dir}")

        target_images = []
        # 1. 遍历图片文件目录
        for img_name in os.listdir(image_dir):
            file_ext = os.path.splitext(img_name)[1]  # 读取后缀.txt

            # 1.1 如果文件后缀无效
            if file_ext not in config.image_extensions:
                continue
            # 1.2 构建image_path
            img_path = str(image_dir / img_name)

            # 1.3 构建图片（上下文）
            img_context = self._find_img_context_with_limit(md_content, img_name, config.img_content_length)

            if not img_context:
                self.logger.warning("MD文件中暂未提取到可用的图片")
                continue  # 继续处理下一个图片文件

            #1.4 提取到当前图片的唯一上下文内容（只选取这张图片第一次出现在文档的位置）
            primary_img_context = img_context[0]
            #1.5 存储到列表中
            target_images.append((img_name, img_path, primary_img_context))

        #1.6返回所有图片完整信息
        self.logger.info(f"找到{len(target_images)}有效的照片")
        return target_images

    def _find_img_context_with_limit(self, md_content: str, img_name: str, max_chars=200 ) ->List[Tuple[str,str,str]]:
        """
        从MD文档中提取图片上下文信息
        思路：使用正则查找图片在md中的位置
        Args:
            md_content:要操作的md文件
            img_name:要定位的图片名字
            max_chars: 最大字符限制
        Returns:
             List[Tuple[str, str, str]]
            List[("离图片最近的上面一个标题","图片的上文","图片的下文")]
        """
        # 1.定义找图片的正则规则（从md找到图片）标准md中的语法结构：![图片的描述](image/aaa.jpg"提示")
        re_patter = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")

        # 2. 按行切分md中的内容，得到一个列表
        md_lines = md_content.split("\n")
        imgs_context = []

        # 3.遍历md
        for line_idx, line in enumerate(md_lines):
            # 3.1 如果当前行不是目标图片就继续查找，
            if not re_patter.search(line):
                continue
            # 3.2.先找离图片最近的标题，从图片的上一行开始查找
            head_title = ""
            head_index = -1
            for i in range(line_idx-1, -1,-1):
                if re.match(r"^#{1,6}\s+", md_lines[i]):
                    head_title = md_lines[i]
                    head_index = i
                    break
            # 寻找图片之前的内容（从标题的下一行到图片的上一行为止）
            pre_content_start_index = head_index + 1
            pre_content = md_lines[pre_content_start_index:line_idx]
            # 3.3 找上文的内容（自下而上，反转）
            img_pre_context = self._extract_img_context_with_limit(pre_content, max_chars, direction="front")

            # 3.4 找下文标题索引
            section_index = len(md_lines)
            for i in range(line_idx+1, section_index):
                if re.match(r"^#{1,6}\s+", md_lines[i]):
                    section_index = i
                    break
            post_content_start_index = line_idx + 1
            post_content = md_lines[post_content_start_index:section_index]
            # 3.5 找下文的内容（正常顺序）
            img_post_context = self._extract_img_context_with_limit(post_content, max_chars, direction="end")
            # 3.6 构建该图片的上下文信息
            imgs_context.append((head_title, img_pre_context, img_post_context))

        # 4.返回该md中当前图片的所有上下文信息（大多数情况下列表只有一个三元组对象)
        return imgs_context
    def _extract_img_context_with_limit(self, extract_content:list, max_chars:int, direction:str)->str:
        """
        提取图片到上下标题（最近）之间的内容（段落）
        想要获取图片“上方”的内容时，保证获取到的是离图片“最近”的那几段。
        md中的段落按照 \n 进行分割，同时行与行之间，每一行后面都有两个空格
        Args:
            extract_content: 提取到的内容
            max_chars:xx
            diretion:

        Returns:

        """

        current_paragraph = []  #存储当前遍历到的内容
        final_paragraph = []    #存储最终遍历到的内容

        # 1.遍历每一行
        for line in extract_content:
            clean_s = line.strip()
            # 如果遇到空行的，就结束
            if not clean_s:
                # 如果当前段落有内容
                if current_paragraph:
                    final_paragraph.append("\n".join(current_paragraph))
                    current_paragraph = []
            else:
                # 如果遇到图片
                if re.match(r"^!\[.*?\]\(.*?\)$", clean_s):
                    if current_paragraph:
                        final_paragraph.append("\n".join(current_paragraph))
                        current_paragraph = []
                    continue
                # 如果不是空白行并且没有遇到图片
                current_paragraph.append(line)

        # 2.处理最后一段。防止current_para中还有一些内容没有处理完成
        # 如果md以文字结尾(或者说最后一段存在且下面没有空行),在上述for循环结束后，最后几行文字还是保存在curr中，没有存进fina。
        if current_paragraph:
            final_paragraph.append("\n".join(current_paragraph))

        # 3.对上文进行处理(从下往上）
        # 要保证对图片上方文字进行提取的时候，一定是最靠近图片的那段文字。因此把离图片最近的段落放在最前面
        if direction == "front":
            final_paragraph.reverse()

        # 4.收集最终返回的段落
        total = 0
        selected = []
        # 对每一行进行判断
        for para in final_paragraph:
            para_len = len(para)
            #如果加上这一段就超字数了，而且已经拿到了至少一段，就停止
            if total + para_len > max_chars and selected:
                break
            selected.append(para)
            total += para_len
        #
        if direction == "front":
            selected.reverse()

        return "\n\n".join(selected)

    def _extract_img_summary(self, document_title: str, target_images_context:List[Tuple[str,str,Tuple[str,str,str]]], config):
        """
        将所有图片生成图片摘要VLM
        Args:
            document_title: 文档名字
            target_images_context: 图片信息
            config:
        Returns:

        """
        self.log_step("step3","准备提取目标摘要")

        summaries = {}
        request_timestamps: Deque[float] = deque()

        # 1. 构建OpenAI客户端
        try:
            client = OpenAI(
                api_key = config.openai_api_key,
                base_url= config.openai_api_base
            )
        except Exception as e:
            logging.error(f"VLM客户端创建失败")
            return summaries

        # 2. 发送请求
        for img_name, img_path, images_context in target_images_context:
            self._enforce_rate_limit(request_timestamps, config.requests_per_minute,60)

            summary = self._get_img_summary(config, client, document_title, img_path, images_context)
            summaries[img_name] = summary

        # 3. 返回映射表sss
        logging.info(f'生成{len(summaries)}图片摘要')
        return summaries

    def _enforce_rate_limit(self, request_timestamps: Deque[float], max_requests: int, window_seconds: int = 60):
        """
        强制执行Api请求速率限制
        Args:
            request_timestamps:
            requests_per_minute:
            window_seconds:

        Returns:

        """
        current_time = time.time()

        # 移除窗口外的时间戳
        while request_timestamps and current_time - request_timestamps[0] >= window_seconds:
            request_timestamps.popleft()

        # 达到上限则等待
        if len(request_timestamps) >= max_requests:
            sleep_duration = window_seconds - (current_time - request_timestamps[0])
            if sleep_duration > 0:
                self.logger.info(f"达到速率限制，暂停 {sleep_duration:.2f} 秒...")
                time.sleep(sleep_duration)

            current_time = time.time()
            while request_timestamps and current_time - request_timestamps[0] >= window_seconds:
                request_timestamps.popleft()

        request_timestamps.append(current_time)

    def _get_img_summary(self, config, client, document_title: str, img_path: str,
                         images_context: Tuple[str, str, str]) -> str:

        # 1.解包images_context构建上下文
        section_title, pre_context, post_context = images_context

        # 2.判断上下文
        context_parts = []
        if section_title:
            context_parts.append(section_title)
        if pre_context:
            context_parts.append(pre_context)
        if post_context:
            context_parts.append(post_context)

        # 3. 构建上下文
        final_context = "\n".join(context_parts) if context_parts else "暂无可用上下文"

        # 4. 读取图片文件
        local_img_content = ""
        try:
            with open(img_path, "rb") as f:
                # 将字节流转化成字符串，最后decode是为了去除前面的b
                local_img_content = base64.b64encode(f.read()).decode("utf-8")
            # 5. 发送请求
        except Exception as e:
            return "暂无图片"

        try:
            response = client.chat.completions.create(
                model=config.vl_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""任务：为Markdown文档中的图片生成一个简短的中文标题。
                                背景信息：
                                    1. 所属文档标题："{document_title}"
                                    2. 图片上下文：{final_context}
                                    请结合图片视觉内容和上述上下文信息，用中文简要总结这张图片的内容，
                                    生成一个精准的中文标题（不要包含"图片"二字）。""",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{local_img_content}"
                                }
                            }
                        ]
                    }
                ]
            )
            summary =response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            self.logger.warning(f"图片摘要生成失败 {img_path} : {e}")
            return "图片描述"

    def _upload_img_and_update_new_md(self,document_name,md_content, images_summmaries, target_images_context,config):
        """
            上传图片到minio以及替换md中的图片url和摘要
        Args:
            md_content:
            image_summmaries:
            target_images_context:

        Returns:

        """
        self.log_step("step4","上传图片到minio并更新md摘要和图片地址")
        remote_urls = {}

        # 1. 构建Minio客户端
        minio_client = get_minio_client()

        if minio_client is None:
            self.logger.warning("无法将本地图片上传到Minio")
        # 2. 遍历图片信息列表
        for img_name, img_path, _ in target_images_context:

            # 2.1 构建对象的名字
            # /1.png
            object_name = f"{document_name}/{img_name}"
            # 2.2 开始上传
            try:
                minio_client.fput_object(
                    config.minio_bucket,
                    object_name,
                    img_path
                )
                # 2.3 手动拼接远程地址
                # http: // 192.168.10.150: 9000 / test - admin /xx .jpg
                remote_url = config.get_minio_base_url() + "/" + config.minio_bucket + "/" + object_name
                self.logger.info(f"{img_name}图片上传到minio成功")
                remote_urls[img_name] = remote_url

            except Exception as e:
                self.logger.warning(f"{img_name}图片上传到minio失败")

        self.logger.info(f"成功上传{len(remote_urls)}张图片到minio")

        # 3.将图片和摘要地址替换到MD内容中去
        new_md_content = md_content

        for img_name, images_summary in images_summmaries.items():

            # 3.1 提取远程地址
            remote_url = remote_urls.get(img_name)
            if not remote_url:
                continue
            # 3.2 替换url和摘要
            replace_pattern = re.compile(
                r"!\[(.*?)\]\((.*?" + re.escape(img_name) + r".*?)\)",
                re.IGNORECASE
            )
            new_md_content = replace_pattern.sub(f"![{images_summary}]({remote_url})", new_md_content)

        return new_md_content




if __name__ == '__main__':
    setup_logging()

    img_md_node = MarkDownImageNode()

    state = {
        'md_path':"D:\A-py\AI_project\smartku\knowledge\processor\import_process\import_temp_dir\万用表RS-12的使用\hybrid_auto\万用表RS-12的使用.md"

    }

    img_md_node.process(state)







