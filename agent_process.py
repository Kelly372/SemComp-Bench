import os
import glob
import io
import math
import base64
import json
import textwrap
from dataclasses import dataclass, field

try:
    import fcntl  # type: ignore
except ImportError:  # Windows does not provide fcntl.
    fcntl = None

import imageio
from PIL import Image, ImageDraw, ImageFont
from volcenginesdkarkruntime import Ark





API_KEY = os.environ.get("VLM_API_KEY", "").strip()
DEFAULT_MODEL = os.environ.get("VLM_MODEL_NAME", "").strip()
DEFAULT_BASE_URL = os.environ.get("VLM_BASE_URL", "").strip()


MODEL_NAME = DEFAULT_MODEL


@dataclass
class VideoSession:





    youtube_url: str
    title: str = ""

    domain: str = ""
    category: str = "Uncertain"

    video_id: str = ""
    file_id: str = ""

    goal: str = ""
    anchor_timestamp: str = ""
    outcome_timestamp: str = ""
    anchor_frame_path: str = ""
    outcome_frame_path: str = ""

    is_blurry: bool = False
    blur_reason: str = ""
    common_components: list = field(default_factory=list)
    instruction_zh: str = ""
    instruction_en: str = ""

    instruction_response_id: str = ""


def set_model_name(model_name: str) -> None:



    global MODEL_NAME
    MODEL_NAME = model_name


def extract_uniform_frames(video_path: str, num_frames: int) -> list[Image.Image]:










    try:
        reader = imageio.get_reader(video_path, format="FFMPEG")
        metadata = reader.get_meta_data()
        fps = metadata.get("fps", 30.0)


        video_duration = metadata.get("duration", 0)
        if video_duration == 0 or video_duration is None:
            try:
                frame_count = reader.count_frames()
                video_duration = frame_count / fps if fps > 0 else 0
            except Exception:
                video_duration = 3600.0

        frames: list[Image.Image] = []


        for i in range(num_frames):

            time_point = (i + 0.5) * video_duration / num_frames


            frame_number = int(time_point * fps)
            if frame_number < 0:
                frame_number = 0

            try:

                frame = reader.get_data(frame_number)
                pil_image = Image.fromarray(frame)
                from media_specs import GRID_CELL_MAX_EDGE, resize_image_max_edge

                frames.append(resize_image_max_edge(pil_image, GRID_CELL_MAX_EDGE))
            except (IndexError, ValueError) as e:
                print(f"警告: 无法读取帧 {frame_number} (时间点 {time_point:.2f}s): {e}")
                continue

        reader.close()
        return frames

    except Exception as e:
        raise ValueError(f"无法读取视频 {video_path}: {e}")


def create_image_grid(frames: list[Image.Image]) -> Image.Image:









    if not frames:
        raise ValueError("帧列表为空")

    num_frames = len(frames)


    if num_frames == 16:
        rows, cols = 4, 4
    elif num_frames == 25:
        rows, cols = 5, 5
    elif num_frames == 36:
        rows, cols = 6, 6
    else:

        cols = int(math.ceil(math.sqrt(num_frames)))
        rows = int(math.ceil(num_frames / cols))


    frame_width, frame_height = frames[0].size


    grid_width = cols * frame_width
    grid_height = rows * frame_height
    grid_image = Image.new("RGB", (grid_width, grid_height), color="black")


    for idx, frame in enumerate(frames):
        row = idx // cols
        col = idx % cols
        x = col * frame_width
        y = row * frame_height
        grid_image.paste(frame, (x, y))

    return grid_image


def image_to_base64(image: Image.Image) -> str:



    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image_base64


def extract_video_id_from_url(youtube_url: str) -> str | None:







    import re

    match = re.search(r"(?:v=|/)([0-9A-Za-z_-]{11}).*", youtube_url)
    if match:
        return match.group(1)
    return None




def load_cache(cache_path: str) -> dict:




    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = json.load(f)
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError) as e:
        print(f"警告: 读取缓存文件 {cache_path} 时出错: {e},返回空缓存")
        return {}


def save_cache(cache_path: str, cache_data: dict):



    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    tmp_path = cache_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        os.replace(tmp_path, cache_path)
    except OSError as e:
        print(f"警告: 保存缓存文件 {cache_path} 时出错: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def update_cache_entry(cache_path: str, youtube_url: str, **fields):



    cache = load_cache(cache_path)
    entry = cache.get(youtube_url, {})
    for key, value in fields.items():
        if value is not None and value != "":
            entry[key] = value
    cache[youtube_url] = entry
    save_cache(cache_path, cache)


def get_cache_entry(cache_path: str, youtube_url: str) -> dict:



    cache = load_cache(cache_path)
    return cache.get(youtube_url, {})


def clear_cache_file_and_response_fields(cache_path: str):






    if not os.path.exists(cache_path):
        print(f"警告: 缓存文件不存在: {cache_path}")
        return
    
    cache = load_cache(cache_path)
    if not cache:

        save_cache(cache_path, {})
        print("缓存文件为空,仅创建空 cache.json")
        return


    fields_to_delete = [
        "stage2_file_id",
        "stage2_response_id",
        "stage3_file_id",
        "stage3_response_id",
        "stage4_file_id",
        "stage4_response_id",
        "stage5_file_id",
        "stage5_response_id",
        "video_clip_file_id",
    ]
    deleted = 0
    total_entries = len(cache)
    
    for entry in cache.values():
        for field in fields_to_delete:
            if field in entry:
                del entry[field]
                deleted += 1

    save_cache(cache_path, cache)
    print(f"已从 {cache_path} 中删除 {deleted} 个 file/response 字段")
    print(f"缓存条目数: {total_entries}")




def extract_frame_at_timestamp(video_path: str, timestamp: str) -> Image.Image:










    try:

        parts = timestamp.split(':')
        if len(parts) == 3:  # HH:MM:SS.f
            hours, minutes, seconds = parts
            total_seconds = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        elif len(parts) == 2:  # MM:SS.f
            minutes, seconds = parts
            total_seconds = float(minutes) * 60 + float(seconds)
        else:  # SS.f
            total_seconds = float(parts[0])
        
        reader = imageio.get_reader(video_path, format='FFMPEG')
        metadata = reader.get_meta_data()
        fps = metadata.get('fps', 30.0)
        

        frame_number = int(total_seconds * fps)
        if frame_number < 0:
            frame_number = 0
        

        frame = reader.get_data(frame_number)
        pil_image = Image.fromarray(frame)
        
        reader.close()
        return pil_image
        
    except Exception as e:
        raise ValueError(f"无法从视频 {video_path} 提取时间戳 {timestamp} 的帧: {e}")


def resize_image_max_edge(image: Image.Image, max_edge: int = 640) -> Image.Image:










    width, height = image.size
    max_dim = max(width, height)
    
    if max_dim == max_edge:
        return image
    

    scale = max_edge / max_dim
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def anchor_frame_disk_path(
    frames_dir: str,
    video_id: str,
    anchor_timestamp: str,
    *,
    for_write: bool = False,
) -> str:




    ts_safe = sanitize_timestamp_for_filename(anchor_timestamp)
    primary = os.path.join(frames_dir, f"{video_id}_{ts_safe}_anchor.jpg")
    if for_write or os.path.isfile(primary):
        return primary
    legacy = os.path.join(frames_dir, f"{video_id}_{ts_safe}_reference.jpg")
    return legacy if os.path.isfile(legacy) else primary


def sanitize_timestamp_for_filename(timestamp: str) -> str:









    if not timestamp:
        return ""

    return timestamp.replace(":", "-")


def parse_timestamp_to_seconds(timestamp: str) -> float:









    if not timestamp:
        return 0.0
    
    try:
        parts = timestamp.split(':')
        if len(parts) == 3:  # HH:MM:SS.f
            hours, minutes, seconds = parts
            return float(hours) * 3600 + float(minutes) * 60 + float(seconds)
        elif len(parts) == 2:  # MM:SS.f
            minutes, seconds = parts
            return float(minutes) * 60 + float(seconds)
        else:  # SS.f
            return float(parts[0])
    except Exception as e:
        print(f"警告: 解析时间戳失败 {timestamp}: {e}")
        return 0.0


def merge_frames_with_instruction(initial_frame_path: str,
                                  end_frame_path: str,
                                  instruction: str,
                                  output_path: str,
                                  image_width: int = 1600,
                                  image_height: int = 900) -> Image.Image:



















    try:
        initial_frame = Image.open(initial_frame_path).convert("RGB")
        end_frame = Image.open(end_frame_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"无法加载帧图片: {e}")
    

    w = (initial_frame.width + end_frame.width) // 2
    h = (initial_frame.height + end_frame.height) // 2
    

    merged_width = int(w * 2.1)
    merged_height = int(h * 1.3)
    

    image_area_height = int(h * 1.1)

    text_area_height = merged_height - image_area_height
    

    frame_area_width = merged_width // 2
    

    image_area = Image.new("RGB", (merged_width, image_area_height), color="white")
    


    initial_x = (frame_area_width - initial_frame.width) // 2
    initial_y = (image_area_height - initial_frame.height) // 2
    image_area.paste(initial_frame, (initial_x, initial_y))
    

    end_x = frame_area_width + (frame_area_width - end_frame.width) // 2
    end_y = (image_area_height - end_frame.height) // 2
    image_area.paste(end_frame, (end_x, end_y))
    

    text_area = Image.new("RGB", (merged_width, text_area_height), color="white")
    draw = ImageDraw.Draw(text_area)
    

    def get_font(size):

        configured_font = os.environ.get("CJK_FONT_PATH", "").strip()
        font_path_candidates = [
            configured_font,
            "NotoSansCJK-Regular.ttc",
            "NotoSansCJK-Bold.ttc",
            "uming.ttc",
            "ukai.ttc",
            "wqy-microhei.ttc",
            "wqy-zenhei.ttc",

            "DejaVuSans.ttf",
            "DejaVuSansMono.ttf",
            "LiberationSans-Regular.ttf",
        ]
        
        for font_path in font_path_candidates:
            if not font_path:
                continue
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
        

        return ImageFont.load_default()
    

    margin = 10
    text_x = margin
    text_y = margin
    max_text_width = merged_width - 2 * margin
    available_height = text_area_height - 2 * margin
    

    if instruction:

        font_size = 14
        font = get_font(font_size)
        line_height = font_size + 4
        

        def wrap_text(text, max_width, font, draw):

            lines = []
            current_line = ""
            

            for char in text:
                test_line = current_line + char
                try:
                    bbox = draw.textbbox((0, 0), test_line, font=font)
                    line_width = bbox[2] - bbox[0]
                except:

                    char_width = font_size * (1.2 if '\u4e00' <= char <= '\u9fff' else 0.6)
                    line_width = len(test_line) * char_width
                
                if line_width <= max_width:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)

                    current_line = char if line_width <= max_width * 2 else ""
            
            if current_line:
                lines.append(current_line)
            
            return lines if lines else [text]
        

        try:
            text_bbox = draw.textbbox((0, 0), instruction, font=font)
            text_width = text_bbox[2] - text_bbox[0]
        except:

            chinese_chars = sum(1 for c in instruction if '\u4e00' <= c <= '\u9fff')
            english_chars = len(instruction) - chinese_chars
            text_width = chinese_chars * font_size * 1.2 + english_chars * font_size * 0.6
        

        if text_width <= max_text_width:
            wrapped_lines = [instruction]
        else:

            wrapped_lines = wrap_text(instruction, max_text_width, font, draw)
        

        title = "Instruction:"
        try:
            title_bbox = draw.textbbox((0, 0), title, font=font)
            title_height = title_bbox[3] - title_bbox[1]
        except:
            title_height = font_size + 2
        
        draw.text((text_x, text_y), title, fill="black", font=font)
        text_y += title_height + 6
        

        for line in wrapped_lines:
            draw.text((text_x, text_y), line, fill="black", font=font)
            text_y += line_height
    else:

        font = get_font(14)
        draw.text((text_x, text_y), "(No instruction)", fill="gray", font=font)
    

    merged_image = Image.new("RGB", (merged_width, merged_height), color="white")
    merged_image.paste(image_area, (0, 0))
    merged_image.paste(text_area, (0, image_area_height))
    

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        merged_image.save(output_path, quality=95)
        print(f"已保存合并图片到: {output_path}")
    
    return merged_image


def translate_instructions_to_chinese(
    json_path: str,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
):













    client = Ark(
        base_url=base_url or DEFAULT_BASE_URL,
        api_key=api_key or API_KEY,
    )
    model = model or DEFAULT_MODEL
    

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON文件不存在: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"JSON文件格式错误: 期望列表,得到 {type(data)}")
    

    translation_prompt = """请将以下英文指令翻译成中文,要求：
1. 保持原意准确
2. 使用自然流畅的中文表达
3. 保留技术术语的准确性
4. 只返回翻译结果,不要添加任何解释或额外内容

英文指令：
{instruction}

中文翻译："""
    
    translated_count = 0
    total_count = len(data)
    
    print(f"开始翻译 {total_count} 条指令...")
    

    for idx, entry in enumerate(data, 1):
        if not isinstance(entry, dict):
            print(f"警告: 条目 {idx} 不是字典格式,跳过")
            continue
        

        if "instruction_zh" in entry and entry["instruction_zh"]:
            print(f"[{idx}/{total_count}] 条目已有翻译,跳过")
            continue
        

        instruction = entry.get("instruction", "")
        if not instruction:
            print(f"[{idx}/{total_count}] 条目缺少instruction字段,跳过")
            continue
        

        try:
            content = [
                {
                    "type": "input_text",
                    "text": translation_prompt.format(instruction=instruction)
                }
            ]
            
            print(f"[{idx}/{total_count}] 正在翻译: {instruction[:50]}...")
            response = client.responses.create(
                model=model,
                input=[
                    {"role": "user", "content": content}
                ]
            )
            

            response_text = None
            

            if hasattr(response, 'output') and isinstance(response.output, list):
                for item in response.output:
                    if hasattr(item, 'content') and isinstance(item.content, list):
                        for content_item in item.content:
                            if hasattr(content_item, 'text'):
                                response_text = content_item.text
                                break
                        if response_text:
                            break

            elif hasattr(response, 'output') and hasattr(response.output, 'choices'):
                response_text = response.output.choices[0].message.content
            elif hasattr(response, 'choices'):
                response_text = response.choices[0].message.content
            
            if not response_text:
                print(f"[{idx}/{total_count}] 警告: VLM响应为空")
                continue
            

            response_text = response_text.strip()
            if response_text.startswith("```"):

                lines = response_text.split("\n")
                lines = [line for line in lines if not line.strip().startswith("```")]
                response_text = "\n".join(lines).strip()
            

            entry["instruction_zh"] = response_text
            translated_count += 1
            
            print(f"[{idx}/{total_count}] 翻译成功: {response_text[:50]}...")
            
        except Exception as e:
            print(f"[{idx}/{total_count}] 翻译失败: {e}")
            continue
    

    print(f"\n翻译完成,共翻译 {translated_count}/{total_count} 条指令")
    print(f"正在保存到 {json_path}...")
    

    tmp_path = json_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, json_path)
        print(f"已成功保存到 {json_path}")
    except Exception as e:
        print(f"保存文件时出错: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
        raise
    
    return translated_count

if __name__ == "__main__":
    translation_json_path = os.environ.get("TRANSLATION_JSON_PATH", "").strip()
    if not translation_json_path:
        raise SystemExit(
            "Set TRANSLATION_JSON_PATH or call translate_instructions_to_chinese(path)."
        )
    translate_instructions_to_chinese(json_path=translation_json_path)

