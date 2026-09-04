import sys

from common.logging.logger import logger, node_log
from shop_brain_graph.import_process.state import ImportGraphState,create_default_state,get_default_state
import requests
from common.config.mineru_config import mineru_config
from pathlib import Path
from rich import print as rprint
from utils.path_util import PROJECT_ROOT
import time

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

    # TODO: 只要取，就校验。
    rprint(f"收到的状态是：\n {state}")


    # TODO: 从state中访问拿到mineru要用的状态
    file_path = state['pdf_path'] if isinstance(state["pdf_path"],Path) else Path(state["pdf_path"])
    file_name = file_path.stem
    
    logger.warning(f"我的主干名是：{file_path}")

    output_dir = state["local_dir"] if isinstance(state["local_dir"],Path) else Path(state["local_dir"])
    # TODO: 调用 MinerU api ，上传文件。
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
    batch_id = None # batch_id 后面获得结果还要用...
    try:
        response = requests.post(url,headers=header,json=data)
        if response.status_code == 200:
            result = response.json()
            print('response success. result:{}'.format(result))
            if result["code"] == 0:
                batch_id = result["data"]["batch_id"]
                urls = result["data"]["file_urls"]
                print('batch_id:{},urls:{}'.format(batch_id, urls))
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
    # TODO 调用Mineru 获得成果文件 轮询直到返回.

    batch_id = batch_id # 放了屁，表示前面已经有了batch_id 
    url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    header = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}"
    }

    interval = 1
    while True :
        res = requests.get(url, headers=header)
        res = res.json()
        if res['data']['extract_result'][0]['state'] == 'done':
            rprint("****"*20)
            rprint("获得结果成功！")
            print(f"{res}")
            rprint("****"*20)
            print(f"获得的下载地址{res['data']['extract_result'][0]['full_zip_url']}")
            rprint("****"*20)
            res = requests.get(res['data']['extract_result'][0]['full_zip_url'])
            with open(output_dir/"output.zip","wb") as f:
                f.write(res.content)
            break
            # TODO: 下载文件，保存到out_dir
            print()
        elif res['data']['extract_result'][0]['state'] == 'failed':
            raise RuntimeError("mineru 解析失败")
        else:
            interval *= 2
            time.sleep(interval)
            continue
    

    # TODO: 运行完添加到done_task
    

    
    return state


if __name__ == "__main__":
    state= create_default_state(pdf_path=PROJECT_ROOT/"doc"/"input"/"hak180产品安全手册.pdf",
                               local_dir=PROJECT_ROOT/"doc"/"output")

    node_pdf_to_md(state)