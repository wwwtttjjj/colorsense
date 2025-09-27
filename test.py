import os
import time
import argparse
import pathlib
import textwrap
import PIL.Image
import google.generativeai as genai

import os
os.environ["HTTP_PROXY"]  = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

keys = [
    "AIzaSyD1-MdBBo8QWuYg8iJ60WMJBIjAH9SRLig",
    "AIzaSyCWWothJ_jkjN9eVoi0gKLxEsdLTLD_D0o",
    "AIzaSyCxSAxHJAjU-HPiiGamvv3YrCmAfSpk5ko",
    "AIzaSyDCaokTY0ApxbjhAbdAe7jvadmydd3m4zA",
    "AIzaSyBeEatZVn-n7TOFexXSuQNk-pEQmoJ8Wac"
]

# ======== 在这里填写你的 API KEY ========
API_KEY = "AIzaSyBeEatZVn-n7TOFexXSuQNk-pEQmoJ8Wac"
genai.configure(api_key=API_KEY)

# 选择模型，例如 gemini-1.5-pro 或 gemini-1.5-flash
model = genai.GenerativeModel("gemini-2.5-flash")

# 发送文本请求
prompt = "用三句话介绍一下深度学习"
response = model.generate_content(prompt)

print("Gemini回答：", response.text)
