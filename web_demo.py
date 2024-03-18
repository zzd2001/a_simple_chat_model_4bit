# 这一版本我们使用gradio构建前端应用，主要原因如下
# 1.gradio相对于streamlit来讲更为简单
# 2.我们正经学过如何使用gradio快速构建机器学习应用的前端界面，这次也正好是一次练习
import gradio as gr
from lmdeploy import turbomind as tm
import torch
import os
import warnings
warnings.filterwarnings('ignore')
# 创建模型存放目录
os.chdir('/home/xlab-app-center')
os.system('mkdir -p model/a_simple_chat_model_w4a16_HF')
# 下载模型
base_path = '/home/xlab-app-center/model/a_simple_chat_model_w4a16_HF'
# download这种方式无效
# from openxlab.model import download
# download(model_repo='Xuanyuan/a_simple_chat_model_w4a16_HF', output=base_path)
# 只能采用官方推荐的这种方式
os.system(f'git clone https://code.openxlab.org.cn/Xuanyuan/a_simple_chat_model.git {base_path}')
os.system(f'cd {base_path} && git lfs pull')
# 查看模型库目录
os.chdir('/home/xlab-app-center/model')
os.system('echo "--/home/xlab-app-center/model--"')
os.system('pwd')
os.system('ls')
os.system('echo "--/home/xlab-app-center/model/--"')

os.chdir('/home/xlab-app-center/model/a_simple_chat_model_w4a16_HF')
os.system('echo "--/home/xlab-app-center/model/a_simple_chat_model_w4a16_HF--"')
os.system('pwd')
os.system('ls')
os.system('echo "--/home/xlab-app-center/model/a_simple_chat_model_w4a16_HF--"')

# # 创建model存放路径
#     os.chdir('/home/xlab-app-center')
#     os.system('mkdir -p model/a_simple_chat_model')  

#     base_path = '/home/xlab-app-center/model/a_simple_chat_model'
#     os.system(f'git clone https://code.openxlab.org.cn/Xuanyuan/a_simple_chat_model.git {base_path}')
#     os.system(f'cd {base_path} && git lfs pull')
#     # # 查看模型库目录
#     # os.chdir('/home/xlab-app-center/model')
#     # os.system('echo "--/home/xlab-app-center/model--"')
#     # os.system('pwd')
#     # os.system('ls')
#     # os.system('echo "--/home/xlab-app-center/model--"')
class ChatModel:
    def __init__(self):
        self.tm_model = tm.TurboMind.from_pretrained(base_path, model_name='internlm-chat-7b')
# 在Python中，以单个下划线（_）开头的方法或变量通常被视为私有的，表示这些方法或变量不应该在类的外部直接访问,自己内部用的
    def _prompt(self,query):
        generator = self.tm_model.create_instance()
        prompt = self.tm_model.model.get_prompt(query)
        input_ids = self.tm_model.tokenizer.encode(prompt)

        for outputs in generator.stream_infer(session_id=0,input_ids=[input_ids]):
            res,tokens = outputs[0]

        response = self.tm_model.tokenizer.decode(res.tolist())
        return response

    def get_response(self,question: str, chat_history: list=[]):
        if question is None or len(question) < 1:
            return "", chat_history
        try:
            question = question.replace(" ",'')
            response = self._prompt(question)
            chat_history.append((question,response))
            return "", chat_history
        except Exception as e:
            return e, chat_history

# 实例化对话model
chat_model = ChatModel()
# 创建gradio接口
block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height = True):
        with gr.Column(scale=15):
            # 展示页面标题
            gr.Markdown("""<h1><center>InternLM Neko Assistant</center></h1>
                            <center>一个基于w4a16量化的intern-chat-7b对话模型</center>
                            """)
    with gr.Row():
        with gr.Column(scale=4):
            # 创建一个chatbot对象
            chatbot = gr.Chatbot(height=450,show_copy_button=True)
            # 创建一个textbox用于输入prompt或者问题
            msg = gr.Textbox(label="Prompt/Question")
            with gr.Row():
                 # 创建一个提交按钮
                db_wo_his_btn = gr.Button("Chat")
            with gr.Row():
                # 创建一个清除的button，用来清除chatbot对话内容
                clear = gr.ClearButton(components=[chatbot],value="Clear console")
        db_wo_his_btn.click(chat_model.get_response,inputs=[msg,chatbot],outputs=[msg,chatbot])  # 第一个参数是一个函数，第二个参数是函数输入参数，第三个参数是函数输出
    gr.Markdown("""Reminder:<br>
        1. Initializing the database may take some time, please be patient.
        2. If any exceptions occur during use, they will be displayed in the text input box. Please do not panic.<br>
        """)
gr.close_all()
# 加载interface
demo.launch()
