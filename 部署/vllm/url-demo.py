import requests
import json
from PIL import Image
import base64
import time
 
def encode_image(image_path):       # 编码本地图片的函数
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
 
start = time.time()
# 1.url
url = 'http://XX.XX.XX.XX:8000/v1/chat/completions'
 
 
# 2.data
## 2.1如果server.py启动，用这个data
data = {"model": "Qwen2-VL-7B",
        "messages": [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},     # 系统命令，一般不要改
                     {"role": "user",
                      "content": "Tell me something about large language models."}],    # 用户命令，一般改这里
        "temperature": 0.7,"top_p": 0.8,"repetition_penalty": 1.05,"max_tokens": 1024}
 
## 2.2使用网络图片(url网址)，用这个data
# data = {"model": "Qwen2-VL-7B",
#         "messages": [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#                      {"role": "user",
#                       "content": [
#                           {"type": "image_url","image_url": {"url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"}},
#                           {"type": "text", "text": "Describe this image."},],}],
#         "temperature": 0.7,"top_p": 0.8,"repetition_penalty": 1.05,"max_tokens": 512}
 
## 2.3使用本地图片，用这个data
    ## 只支持一张图片，可以进行OCR、翻译、计算题目、编写前端代码等
# image_path = "jieti.jpg"
# base64_image = encode_image(image_path)
# data = {"model": "Qwen2-VL-7B",
#         "messages": [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#                      {"role": "user",
#                       "content": [
#                           {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
#                           {"type": "text", "text": "解一下这道题"},],}],
#         "temperature": 0.7,"top_p": 0.8,"repetition_penalty": 1.05,"max_tokens": 1024}
 
## 2.4使用本地图片，多张照片理解
# image_path1 = "1.jpg"
# image_path2 = "2.jpg"
# base64_image1 = encode_image(image_path1)
# base64_image2 = encode_image(image_path2)
# data = {"model": "Qwen2-VL-7B",
#         "messages":[
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_image1}"
#                     },
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_image2}"
#                     },
#                 },
#                 {"type": "text", "text": "这些是什么"},
#             ],}
#         ],
#         "temperature": 0.7,"top_p": 0.8,"repetition_penalty": 1.05,"max_tokens": 4096}
 
 
# 3.将字典转换为 JSON 字符串
json_payload = json.dumps(data)
 
 
# 4.发送 POST 请求
headers = {'Content-Type': 'application/json'}
response = requests.post(url, data=json_payload, headers=headers)
 
 
# 5.打印响应内容
print(response.json().get("choices", [])[0].get("message", []).get("content", []))
# print(response.json())        # 调试用
 
 
print("\n总时间：", time.time()-start, "秒")
 