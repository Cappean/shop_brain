import sys

from common.logging.logger import logger, node_log
from shop_brain_graph.import_process.state import ImportGraphState,create_default_state,get_default_state
import requests
from common.config.mineru_config import mineru_config
from pathlib import Path

@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    节点: PDF转Markdown (node_pdf_to_md)
    为什么叫这个名字: 核心任务是将 PDF 非结构化数据转换为 Markdown 结构化数据。
    未来要实现:
    1. 调用 MinerU (magic-pdf) 工具。
    2. 将 PDF 转换成 Markdown 格式。
    3. 将结果保存到 state["md_content"]。
    """

    # TODO: 依旧添加到running_tasks 前端要用

    # TODO: 从state中访问拿到mineru要用的状态
    file_path = state["local_file_path"] if isinstance(state["local_file_path"],Path) else Path(state["local_file_path"])
    file_name = file_path.stem

    # TODO: 调用 MinerU (magic-pdf) 工具。
    token = mineru_config.api_key
    url = "https://mineru.net/api/v4/file-urls/batch"
    header = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "files": [
            {"name":file_name, "data_id": "1"}
        ],
        "model_version":"vlm"
    }
    file_path = [str(file_path)]
    try:
        response = requests.post(url,headers=header,json=data)
        if response.status_code == 200:
            result = response.json()
            # print('response success. result:{}'.format(result))
            if result["code"] == 0:
                batch_id = result["data"]["batch_id"]
                urls = result["data"]["file_urls"]
                # print('batch_id:{},urls:{}'.format(batch_id, urls))
                for i in range(0, len(urls)):
                    with open(file_path[i], 'rb') as f:
                        res_upload = requests.put(urls[i], data=f)
                        if res_upload.status_code == 200:
                            print(f"{urls[i]} upload success")
                        else:
                            print(f"{urls[i]} upload failed")
            else:
                print('apply upload url failed,reason:{}'.format(result["msg"]))
        else:
            print('response not success. status:{} ,result:{}'.format(response.status_code, response))
    except Exception as err:
        print(err)

    # TODO: 运行完添加到done_task
    

    
    return state