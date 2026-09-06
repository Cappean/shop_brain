import base64
from mimetypes import guess_type
import sys

from common.logging.logger import logger, node_log
from shop_brain_graph.import_process.state import ImportGraphState, create_default_state,get_default_state
from utils.task_utils import add_running_task,add_done_task
from utils.lm.lm_utils import get_llm_client
from common.config.lm_config import lm_config
from utils.load_prompt import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain.messages import HumanMessage

from pathlib import Path
import re

@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 图片处理 (node_md_img)
    为什么叫这个名字: 处理 Markdown 中的图片资源 (Image)。
    未来要实现:
    1. 扫描 Markdown 中的图片链接。
    2. 将图片上传到 MinIO 对象存储。
    3. (可选) 调用多模态模型生成图片描述。
    4. 替换 Markdown 中的图片链接为 MinIO URL。

    每个节点默写一遍state :
        task_id 

        pdf_enable
        md_enable 
        pdf_path
        md_path
        md_content 

        local_file_path : 这个是第一个节点node_entry要用到的一开始传进来的文件路径，还没有分类pdf 或 md 
        local_dir : 这个是第二个节点 node_pdf_to_md 用到的，用来存放pdf使用mineru转成的zip的文件夹路径，也就是输出路径。

        file_title : 文件名？ 
        .... 
    """
    # TODO 添加到前端要用的运行中节点
    add_running_task(state["task_id"],"node_md_img")
    # TODO 从文档中读取内容并校验 ： 
    # 读取md_path来获得文件名以及md的内容，
    # 同时读取local_dir来获得输出路径，拼上文件名+images就是所有的图片所在的路径。
    
    # 检查状态是否有内容
    if (not state["md_path"]) or (not state["file_title"]) or (not state["local_dir"]) :
        raise RuntimeError("错误，节点参数不全，请检查。")
    
    md_path = state["md_path"] if  isinstance(state["md_path"],Path) else Path(state["md_path"])
    file_name = state["file_title"]
    local_dir = state["local_dir"] if  isinstance(state["local_dir"],Path) else Path(state["local_dir"])
    images_dir = local_dir / file_name /"images"

    # 检查路径对应实体是否存在。
    if not md_path.exists() :
        logger.error(f"错误，md_path不存在，请检查。{md_path}")
        raise FileNotFoundError(f"错误，md_path不存在，请检查。{md_path}")
    
    if not images_dir.exists() :
        logger.warning(f"路径：{md_path } 对应md 文档无images文件夹，请核实md文件内是否存在图片。")
        return state

    # TODO 从文档中读取内容并校验 ：
    state["md_content"] = md_path.read_text(encoding="utf-8")
    
    # TODO 遍历图片所在的目录，对于每一张图片，
    # 1拿到图片名，
    # 2然后它的路径，
    # 3用正则去匹配md 文档里面的内容。去得到上下文。
    # 最后放到一个列表里面 
    images_info_list = []
    for image in images_dir.iterdir():
        image_name = image.name
        search_target = re.escape(image_name) # 使用 re.escape() 函数来转义特殊字符

        # 正则匹配md文档中的格式: ![alt text](/path/to/image.png)
        pattern = r'!\[(.*?)\]\(([^)]*' + search_target + r'[^)]*)\)'
        match = re.search(pattern, state["md_content"])
        if match :
            logger.info(f"用正则匹配到的match对象是: {match}")
        else : 
            logger.warning("该图片未出现在md中")
            continue # 因为图片不存在，一旦放过去，那么就会有空指针。

        pre_context = state["md_content"][max(0,match.start()-100):match.start()]
        post_context = state["md_content"][match.end():min(len(state["md_content"]),match.end()+100)]
        logger.info(f"这是这个图片匹配到的上文{pre_context}，\n这是这个图片匹配到的下文{post_context}")

    # TODO 然后1.文件名，2.图片 3.上下文组合生成提示词，调用视觉模型获得摘要
        vision_lm = get_llm_client(lm_config.vl_model)
        image_content = (pre_context,post_context)
        prompt = load_prompt("image_summary",root_folder = state["file_title"],image_content = image_content)
        logger.info(f"这是这个图片的prompt: {prompt}")

        stroutParser = StrOutputParser()
        chain = vision_lm | stroutParser 

        image_base64_str = base64.b64encode(image.read_bytes()).decode('utf-8')
        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{guess_type(image_name)[0]};base64,{image_base64_str}"}
                },  # 图片
                {
                    "type": "text", "text": prompt
                }  # 文本
            ]
        )
        summary_result = chain.invoke([message])
        
        logger.error(f"这是这个图片的摘要: {summary_result}")


    # TODO 然后将获得的摘要用正则替换掉md_content 中的图片的内容。替换图片的摘要部分就完成了。

    # TODO 然后将图片上传到MINIO，获得图片的url，用正则替换掉md_content 中的图片的url部分。替换图片的url就完成了。

    # TODO 最后将修改后的md_content 写入到md_path旁边，就是在原来的文件名字后面加：_processed.md。完成图片处理。

    # TODO 添加到前端要用的运行完成节点
    add_done_task(state["task_id"],"node_md_img")

    return state


if __name__ == "__main__":
    state = create_default_state(
        task_id = "12344455",
        local_dir = r"E:\UV_LLM_progrems\shop_brain\doc\output",
        md_path = r"E:\UV_LLM_progrems\shop_brain\doc\output\hak180产品安全手册\hak180产品安全手册.md",
        file_title = "hak180产品安全手册"
    )
    node_md_img(state)