import boto3
import json
import re
from PIL import Image, ImageDraw, ImageFont
import io

# from nova_image_grounding_demo import print_messages_without_images

# 初始化Amazon Bedrock客户端
modelId = "us.amazon.nova-lite-v1:0"  # 默认使用nova-lite-v1模型
accept = "application/json"
contentType = "application/json"
def print_messages_without_images(messages):
    """打印messages的树形结构"""
    def print_tree(obj, prefix="", is_last=True, indent="  "):
        # 打印当前节点
        branch = "└── " if is_last else "├── "
        print(f"{prefix}{branch}", end="")
        
        if isinstance(obj, dict):
            if "image" in obj:
                print("<image>")
            elif "source" in obj and "data" in obj["source"]:
                print("<image_data>")
            else:
                print("{")
                new_prefix = prefix + (indent if is_last else "│   ")
                items = list(obj.items())
                for i, (key, value) in enumerate(items):
                    print(f"{new_prefix}├── {key}: ", end="")
                    if isinstance(value, (dict, list)):
                        print()
                        print_tree(value, new_prefix + indent, i == len(items) - 1)
                    else:
                        if isinstance(value, str) and len(value) > 50:
                            print(f"{value[:50]}...")
                        else:
                            print(value)
                if not items:
                    print("}")
        elif isinstance(obj, list):
            print("[")
            new_prefix = prefix + (indent if is_last else "│   ")
            for i, item in enumerate(obj):
                print_tree(item, new_prefix, i == len(obj) - 1)
            if not obj:
                print(f"{prefix}{indent}]")
        else:
            if isinstance(obj, str) and len(obj) > 50:
                print(f"{obj[:50]}...")
            else:
                print(obj)

    print("\nMessages Tree Structure:")
    print_tree(messages)
try:
    bedrock_rt = boto3.client("bedrock-runtime", region_name="us-east-1")
except Exception as e:
    print(f"初始化Bedrock客户端时出错: {e}")
    print("请确保您已配置AWS凭证")

def safe_json_load(json_string):
    try:
        print("原始输入:", json_string)
        
        # 移除所有空白字符
        json_string = re.sub(r"\s", "", json_string)
        print("移除空白后:", json_string)
        
        # 尝试直接解析JSON
        try:
            result = json.loads(json_string)
            if isinstance(result, dict):
                return result
        except:
            pass
        
        # 如果直接解析失败，尝试提取边界框信息
        objects = []
        # 匹配所有可能的边界框格式
        patterns = [
            r'\{"([^"]+)":\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]\}',  # 完整对象格式
            r'"([^"]+)":\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'       # 简单格式
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, json_string)
            for match in matches:
                label = match.group(1)
                coords = [int(match.group(i)) for i in range(2, 6)]
                objects.append({label: coords})
                print(f"找到对象: {label}, 坐标: {coords}")
        
        # 返回解析结果
        result = {
            "objects": objects,
            "object_count": 0  # 默认值为0，保持原始返回的object_count
        }
        return result
        
    except Exception as e:
        print("原始JSON字符串:", json_string)
        print("解析错误:", e)
        # 返回默认结构
        return {"objects": [], "object_count": 0}

def detect_objects(image_pil, image_short_size, detection_prompt, category, reference_images=None, latency_mode='standard', draw_bbox=True):
    """
    使用Nova模型检测图像中的对象
    
    Args:
        image_pil: PIL图像对象
        image_short_size: 图像短边大小
        detection_prompt: 检测提示词
        category: 检测类别
        reference_images: 参考图片列表
        latency_mode: 延迟模式
        
    Returns:
        tuple: (处理后的图像, 日志输出, 性能指标)
    """
    try:
        width, height = image_pil.size
        
        ratio = image_short_size / min(width, height)
        width = round(ratio * width)
        height = round(ratio * height)
        
        image_pil = image_pil.resize((width, height), resample=Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image_pil.save(buffer, format="webp", quality=90)
        image_data = buffer.getvalue()
    
        # 根据是否需要绘制边界框选择不同的prefill
        prefill='{'
        
        # 构建消息序列
        messages = []
        
        # 如果有参考图片，添加完整的参考图片部分
        if reference_images:
            messages.extend([
                {
                    "role": "user",
                    "content": [{
                        "text": "The following images are EXAMPLES. Do not analyze these, only use them for reference."
                    }]
                },
                {
                    "role": "assistant",
                    "content": [{
                        "text": "Understood. I will not analyze these example images, only use them as reference. Please send them to me."
                    }]
                }
            ])
            
            example_content = [{
                "text": "<!-- These are example images of the type of element you are looking for. Use only as guidance.--> <example_images>"
            }]

            for ref_img in reference_images:
                # 处理参考图片
                ref_width, ref_height = ref_img.size
                ref_ratio = image_short_size / min(ref_width, ref_height)
                ref_width = round(ref_ratio * ref_width)
                ref_height = round(ref_ratio * ref_height)
                ref_img = ref_img.resize((ref_width, ref_height), resample=Image.Resampling.LANCZOS)
                
                ref_buffer = io.BytesIO()
                ref_img.save(ref_buffer, format="webp", quality=90)
                ref_image_data = ref_buffer.getvalue()
                
                example_content.extend([
                    {
                        "text": "<!-- This is an example image of the type of element you are looking for. Use only as guidance.--> <example_image>"
                    },
                    {
                        "image": {
                            "format": "webp",
                            "source": {
                                "bytes": ref_image_data
                            }
                        }
                    },
                    {
                        "text": "</example_image>"
                    }
                ])

            example_content.append({
                "text": "</example_images> <!--That is all the example images I have for you.-->"
            })

            messages.extend([
                {
                    "role": "user",
                    "content": example_content
                },
                {
                    "role": "assistant",
                    "content": [{
                        "text": "I've received the example images and will use them only as reference."
                    }]
                }
            ])

        # 如果有参考图片，需要添加过渡提示
        if reference_images:
            messages.extend([
                {
                    "role": "user",
                    "content": [{
                        "text": "The next image I send will be the ONLY image for analysis. Disregard all previous images for analysis purposes."
                    }]
                },
                {
                    "role": "assistant",
                    "content": [{
                        "text": "Understood. I will only analyze the next image you send, disregarding all previous images for analysis."
                    }]
                }
            ])
        
        # 添加待分析图片和检测指令
        messages.extend([
            {
                "role": "user",
                "content": [
                    {
                        "text": "<!-- This is the image you will use for your analysis tasks --> <image_to_analyze>"
                    },
                    {
                        "image": {
                            "format": "webp",
                            "source": {
                                "bytes": image_data
                            }
                        }
                    },
                    {
                        "text": "<!-- That was the image you will use for your analysis tasks. --> </image_to_analyze>"
                    }
                ]
            },
            {
                "role": "user",
                "content": [{
                    "text": detection_prompt
                }]
            },
            {
                "role": "assistant",
                "content": [{
                    "text": prefill
                }]
            }
        ])
        print_messages_without_images(messages)
        # 调用模型
        response = bedrock_rt.converse(
            modelId=modelId, 
            messages=messages,
            inferenceConfig={
                "temperature": 0.0,
                "maxTokens": 1024,
            },
            performanceConfig={
                'latency': latency_mode
            }
        )
        model_output = response.get('output')["message"]["content"][0]["text"];
        print(f"model output \n  {model_output}")
        output = prefill + model_output
        print("-----")
        print(output)
        print("-------")
        result = safe_json_load(output)
        print(result)
        print("hello ----")
        log_output = []
        
        # 获取对象计数
        object_count = result.get("object_count", 0)
        
        # 添加对象计数日志
        if object_count > 0:
            log_output.append(f"检测到{object_count}个目标对象")
        else:
            log_output.append("未检测到目标对象")
            
        # 绘制结果
        color_list = [
            'blue',
            'green',
            'yellow',
            'red',
            'orange',
            'pink',
            'purple',
        ]
        
        try:
            font = ImageFont.truetype("Arial", size=height // 20)
            count_font = ImageFont.truetype("Arial", size=26)
        except IOError:
            # 如果找不到Arial字体，使用默认字体
            font = ImageFont.load_default()
            count_font = font
        
        # 在右上角绘制对象计数
        draw = ImageDraw.Draw(image_pil)
        count_text = f"Objects: {object_count}"
        # 获取文本大小
        if isinstance(count_font, ImageFont.FreeTypeFont):
            text_bbox = count_font.getbbox(count_text)
            text_width = text_bbox[2] - text_bbox[0]
        else:
            text_width = count_font.getsize(count_text)[0]
        # 计算文本位置（右上角，留出10像素边距）
        text_x = width - text_width - 10
        text_y = 10
        # 绘制文本（白色文字带黑色描边以确保在任何背景下都清晰可见）
        for offset in [(-1,-1), (-1,1), (1,-1), (1,1)]:  # 描边
            draw.text((text_x + offset[0], text_y + offset[1]), 
                    count_text, font=count_font, fill='black')
        draw.text((text_x, text_y), count_text, font=count_font, fill='white')
        
        for idx, item in enumerate(result.get("objects", [])):
            label = next(iter(item)).strip()
            if label == "others" or label == "other":
                continue
            bbox = item[label]
            x1, y1, x2, y2 = bbox
            
            if x1 >= x2 or y1 >= y2:
                continue
                
            w, h = image_pil.size
            x1 = x1 / 1000 * w
            x2 = x2 / 1000 * w
            y1 = y1 / 1000 * h
            y2 = y2 / 1000 * h
            
            bbox = (x1, y1, x2, y2)
            bbox = list(map(round, bbox))
            
            log_message = f"检测到 <{label}> 在坐标 {bbox}"
            log_output.append(log_message)
            
            print(f"检测到标签: {label}")
            
            draw = ImageDraw.Draw(image_pil)
            color = color_list[idx % len(color_list)]
            draw.rectangle(bbox, outline=color, width=2)
            draw.text((x1 + 4, y1 + 2), label, fill=color, font=font)
        
        # 获取性能指标和使用统计
        metrics = response.get('metrics', {})
        latency_ms = metrics.get('latencyMs', 'N/A')
        performance_config = response.get('performanceConfig', {})
        usage = response.get('usage', {})
        
        metrics_log = (
            f"\n性能指标"
            f"\n延迟：{latency_ms}ms"
            f"\n性能配置：{performance_config}"
            f"\nToken使用统计"
            f"\n输入tokens：{usage.get('inputTokens', 'N/A')}"
            f"\n输出tokens：{usage.get('outputTokens', 'N/A')}"
            f"\n总tokens：{usage.get('totalTokens', 'N/A')}"
        )
        log_output.append(metrics_log)
        
        # 构建metrics对象
        metrics_obj = ["N/A", metrics.get('latencyMs', 'N/A'), usage.get('inputTokens', 'N/A'), usage.get('outputTokens', 'N/A')]
        return image_pil, "\n".join(log_output), metrics_obj
    
    except Exception as e:
        print(f"处理单张图片时出错: {str(e)}")
        return None, None, None
