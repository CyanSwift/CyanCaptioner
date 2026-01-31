"""
在线视频字幕生成系统
使用多模态大模型分析视频事件并生成字幕
"""

import os
import json
import uuid
import base64
import requests
import cv2
import numpy as np
import asyncio
import edge_tts
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB限制

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# API配置
API_URL = "https://api-inference.modelscope.cn/v1/chat/completions"
API_KEY = os.environ.get("MODELSCOPE_API_KEY", "")
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def calculate_frame_difference(frame1, frame2):
    """计算两帧之间的差异度（0-1）"""
    # 转换为灰度图
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # 计算绝对差异
    diff = cv2.absdiff(gray1, gray2)
    
    # 计算差异比例
    diff_ratio = np.sum(diff > 30) / diff.size
    
    return diff_ratio

def calculate_histogram_difference(frame1, frame2):
    """计算两帧直方图的差异度（0-1）"""
    # 转换为HSV
    hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
    
    # 计算直方图
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [50, 60], [0, 180, 0, 256])
    
    # 归一化
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
    # 比较直方图（值越小越相似）
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    return 1 - similarity  # 转换为差异度

def extract_frames_by_content(video_path, threshold=0.15, min_interval=1.0, max_interval=10.0):
    """基于内容变化提取关键帧
    
    Args:
        video_path: 视频路径
        threshold: 变化阈值（0-1），越小越敏感
        min_interval: 最小采样间隔（秒）
        max_interval: 最大采样间隔（秒），即使没有变化也会采样
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception("无法打开视频文件")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    min_frame_interval = int(fps * min_interval)
    max_frame_interval = int(fps * max_interval)
    
    # 读取第一帧
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return frames, duration
    
    # 添加第一帧
    _, buffer = cv2.imencode('.jpg', prev_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    frames.append({
        'timestamp': 0,
        'frame_base64': base64.b64encode(buffer).decode('utf-8'),
        'change_type': 'start'
    })
    
    last_keyframe_idx = 0
    frame_idx = 1
    
    while True:
        ret, current_frame = cap.read()
        if not ret:
            break
        
        frames_since_last = frame_idx - last_keyframe_idx
        
        # 检查是否达到最小间隔
        if frames_since_last >= min_frame_interval:
            # 计算与上一关键帧的差异
            pixel_diff = calculate_frame_difference(prev_frame, current_frame)
            hist_diff = calculate_histogram_difference(prev_frame, current_frame)
            
            # 综合差异度
            combined_diff = 0.6 * pixel_diff + 0.4 * hist_diff
            
            # 判断是否为关键帧
            is_keyframe = False
            change_type = None
            
            if combined_diff > threshold:
                is_keyframe = True
                if hist_diff > 0.5:
                    change_type = 'scene_change'  # 场景切换
                else:
                    change_type = 'content_change'  # 内容变化
            elif frames_since_last >= max_frame_interval:
                is_keyframe = True
                change_type = 'interval'  # 强制采样
            
            if is_keyframe:
                timestamp = frame_idx / fps
                _, buffer = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frames.append({
                    'timestamp': timestamp,
                    'frame_base64': base64.b64encode(buffer).decode('utf-8'),
                    'change_type': change_type,
                    'diff_score': combined_diff
                })
                prev_frame = current_frame.copy()
                last_keyframe_idx = frame_idx
        
        frame_idx += 1
    
    cap.release()
    return frames, duration

def extract_frames(video_path, interval_seconds=2):
    """从视频中按时间间隔提取帧"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception("无法打开视频文件")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    frame_interval = int(fps * interval_seconds)
    current_frame = 0
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        timestamp = current_frame / fps
        # 将帧转换为base64
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        frames.append({
            'timestamp': timestamp,
            'frame_base64': frame_base64
        })
        
        current_frame += frame_interval
        if current_frame >= total_frames:
            break
    
    cap.release()
    return frames, duration

def call_multimodal_api(frame_base64, prompt, context=None, max_retries=3):
    """调用多模态大模型API
    
    Args:
        frame_base64: 图片的base64编码
        prompt: 提示词
        context: 上下文信息（之前的字幕描述）
        max_retries: 最大重试次数
    """
    # 检查 API_KEY 是否配置
    if not API_KEY:
        raise Exception("API_KEY 未配置，请设置环境变量 MODELSCOPE_API_KEY")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # 构建消息内容
    content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame_base64}"
            }
        }
    ]
    
    # 如果有上下文，加入到提示中
    if context:
        full_prompt = f"""上下文（之前的画面描述）：
{context}

---
{prompt}"""
    else:
        full_prompt = prompt
    
    content.append({
        "type": "text",
        "text": full_prompt
    })
    
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": content
        }],
        "max_tokens": 1024,
        "temperature": 0.7,
        "enable_thinking": False
    }
    
    # 增加重试机制
    import time
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            
            if response.status_code == 200:
                # 检查响应是否为 JSON
                content_type = response.headers.get('Content-Type', '')
                if 'application/json' not in content_type:
                    last_error = f"API 返回非 JSON 响应: {response.text[:200]}"
                    print(f"尝试 {attempt + 1}/{max_retries}: {last_error}")
                    time.sleep(1)
                    continue
                result = response.json()
                return result['choices'][0]['message']['content']
            elif response.status_code == 429:
                # 限流，等待后重试
                print(f"API 限流，等待 {2 ** attempt} 秒后重试...")
                time.sleep(2 ** attempt)
                continue
            elif response.status_code == 401:
                raise Exception("API_KEY 无效或已过期，请检查环境变量 MODELSCOPE_API_KEY")
            else:
                last_error = f"API调用失败: {response.status_code} - {response.text[:200]}"
                print(f"尝试 {attempt + 1}/{max_retries}: {last_error}")
                time.sleep(1)
        except requests.exceptions.Timeout:
            last_error = "API 调用超时"
            print(f"尝试 {attempt + 1}/{max_retries}: {last_error}")
            time.sleep(1)
        except Exception as e:
            last_error = str(e)
            print(f"尝试 {attempt + 1}/{max_retries}: {last_error}")
            if "API_KEY" in last_error:
                raise  # 直接抛出 API_KEY 相关错误
            time.sleep(1)
    
    raise Exception(f"API调用失败（重试{max_retries}次）: {last_error}")

def analyze_video_events(video_path, interval_seconds=2, detection_mode='interval', threshold=0.15, language='zh'):
    """分析视频中的事件
    
    Args:
        video_path: 视频路径
        interval_seconds: 固定间隔模式的采样间隔
        detection_mode: 'interval' 固定间隔 | 'content' 内容变化检测
        threshold: 内容变化检测的阈值
        language: 字幕语言 'zh' 中文 | 'en' 英文
    """
    if detection_mode == 'content':
        frames, duration = extract_frames_by_content(
            video_path, 
            threshold=threshold,
            min_interval=1.0,
            max_interval=interval_seconds
        )
    else:
        frames, duration = extract_frames(video_path, interval_seconds)
    
    subtitles = []
    
    # 根据语言选择prompt
    if language == 'en':
        prompt = """Please carefully observe this image and describe the main event or action happening in the scene.
Requirements:
1. Use concise English description (no more than 50 words)
2. Only describe the most prominent event
3. If no obvious event, describe the scene state
4. Output the description directly without any prefix or explanation
5. If context is provided, maintain coherence, avoid repeating previously described content, focus on changes or new events"""
    else:
        prompt = """请仔细观察这张图片，描述画面中正在发生的主要事件或动作。
要求：
1. 用简洁的中文描述（不超过30字）
2. 只描述最主要、最显著的事件
3. 如果画面中没有明显事件，描述场景状态
4. 直接输出描述内容，不要添加任何前缀或解释
5. 如果提供了上下文，请保持描述的连贯性，避免重复之前已描述过的内容，着重描述变化或新发生的事件"""

    # 维护最近的上下文（最多3条）
    context_window = []
    max_context = 3

    for i, frame_data in enumerate(frames):
        try:
            # 构建上下文字符串
            context = None
            if context_window:
                context = "\n".join([f"[{c['time']}] {c['text']}" for c in context_window])
            
            description = call_multimodal_api(frame_data['frame_base64'], prompt, context)
            # 清理描述文本
            description = description.strip()
            if description:
                # 计算结束时间
                if i + 1 < len(frames):
                    end_time = frames[i + 1]['timestamp']
                else:
                    end_time = duration
                
                subtitle_entry = {
                    'id': i + 1,
                    'start_time': frame_data['timestamp'],
                    'end_time': end_time,
                    'text': description
                }
                
                # 添加变化类型信息（内容检测模式）
                if 'change_type' in frame_data:
                    subtitle_entry['change_type'] = frame_data['change_type']
                
                subtitles.append(subtitle_entry)
                
                # 更新上下文窗口
                context_window.append({
                    'time': f"{int(frame_data['timestamp']//60):02d}:{int(frame_data['timestamp']%60):02d}",
                    'text': description
                })
                if len(context_window) > max_context:
                    context_window.pop(0)
                    
        except Exception as e:
            print(f"分析第{i+1}帧时出错: {str(e)}")
            continue
    
    return subtitles, duration

def format_time_srt(seconds):
    """将秒数转换为SRT时间格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def generate_srt(subtitles):
    """生成SRT格式字幕"""
    srt_content = ""
    for sub in subtitles:
        srt_content += f"{sub['id']}\n"
        srt_content += f"{format_time_srt(sub['start_time'])} --> {format_time_srt(sub['end_time'])}\n"
        srt_content += f"{sub['text']}\n\n"
    return srt_content

def generate_csv(subtitles):
    """生成CSV格式字幕"""
    import csv
    import io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['id', 'start_time', 'end_time', 'text'])
    for sub in subtitles:
        writer.writerow([sub['id'], sub['start_time'], sub['end_time'], sub['text']])
    return output.getvalue()

def burn_subtitles(video_path, subtitles, output_path):
    """将字幕烧录到视频中"""
    cap = cv2.VideoCapture(video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = frame_idx / fps
        
        # 查找当前时间的字幕
        current_subtitle = None
        for sub in subtitles:
            if sub['start_time'] <= current_time < sub['end_time']:
                current_subtitle = sub['text']
                break
        
        if current_subtitle:
            # 添加字幕背景和文字
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            
            # 计算文字大小
            text = current_subtitle
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # 字幕位置（底部居中）
            x = (width - text_width) // 2
            y = height - 50
            
            # 绘制半透明背景
            overlay = frame.copy()
            cv2.rectangle(overlay, (x - 10, y - text_height - 10), 
                         (x + text_width + 10, y + baseline + 10), 
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # 绘制文字
            cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    
    return output_path


# TTS 音色配置
TTS_VOICES = {
    'zh': {
        'female': 'zh-CN-XiaoxiaoNeural',
        'male': 'zh-CN-YunxiNeural'
    },
    'en': {
        'female': 'en-US-JennyNeural',
        'male': 'en-US-GuyNeural'
    }
}


async def generate_single_tts(text, voice, output_path):
    """生成单个 TTS 音频文件"""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    return output_path


def generate_tts_for_subtitles(subtitles, language='zh', voice_gender='female', output_dir='temp_audio'):
    """为字幕生成 TTS 音频
    
    Args:
        subtitles: 字幕列表
        language: 语言 'zh' 或 'en'
        voice_gender: 音色 'female' 或 'male'
        output_dir: 输出目录
    
    Returns:
        audio_segments: 包含音频路径和时间信息的列表
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 选择音色
    voice = TTS_VOICES.get(language, TTS_VOICES['zh']).get(voice_gender, 'zh-CN-XiaoxiaoNeural')
    
    audio_segments = []
    
    # 使用 asyncio 生成 TTS
    async def generate_all():
        for i, sub in enumerate(subtitles):
            audio_path = os.path.join(output_dir, f"tts_{i}.mp3")
            try:
                await generate_single_tts(sub['text'], voice, audio_path)
                audio_segments.append({
                    'id': sub['id'],
                    'audio_path': audio_path,
                    'original_start': sub['start_time'],
                    'original_end': sub['end_time'],
                    'text': sub['text']
                })
            except Exception as e:
                print(f"生成第 {i+1} 条 TTS 失败: {str(e)}")
                continue
    
    # 运行异步任务
    asyncio.run(generate_all())
    
    return audio_segments


def get_audio_duration(audio_path):
    """获取音频文件时长"""
    from moviepy.editor import AudioFileClip
    try:
        clip = AudioFileClip(audio_path)
        duration = clip.duration
        clip.close()
        return duration
    except Exception as e:
        print(f"获取音频时长失败: {str(e)}")
        return 3.0  # 默认返回3秒


def calculate_audio_timing(audio_segments):
    """计算音频时间轴，处理重叠问题
    
    策略：如果前一段音频还未播放完，跳过当前音频不播报
    """
    timing_info = []
    skipped_info = []
    current_end_time = 0
    
    for seg in audio_segments:
        duration = get_audio_duration(seg['audio_path'])
        original_start = seg['original_start']
        
        # 如果原始开始时间早于当前结束时间，跳过这条配音
        if original_start < current_end_time:
            skipped_info.append({
                'id': seg['id'],
                'original_start': original_start,
                'reason': f'与上一条配音冲突（上一条结束于 {current_end_time:.2f}s）',
                'text': seg['text']
            })
            print(f"跳过配音 #{seg['id']}: 时间冲突，原始开始时间 {original_start:.2f}s < 当前结束时间 {current_end_time:.2f}s")
            continue
        
        actual_start = original_start
        actual_end = actual_start + duration
        current_end_time = actual_end
        
        timing_info.append({
            'id': seg['id'],
            'audio_path': seg['audio_path'],
            'original_start': original_start,
            'actual_start': actual_start,
            'actual_end': actual_end,
            'duration': duration,
            'adjusted': False,
            'text': seg['text']
        })
    
    if skipped_info:
        print(f"共跳过 {len(skipped_info)} 条配音因时间冲突")
    
    return timing_info


def merge_audio_with_video(video_path, audio_timing, output_path, keep_original_audio=True):
    """将 TTS 音频合成到视频中
    
    Args:
        video_path: 原视频路径
        audio_timing: 音频时间信息列表
        output_path: 输出路径
        keep_original_audio: 是否保留原声
    """
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
    
    video = VideoFileClip(video_path)
    video_duration = video.duration
    
    # 创建 TTS 音频片段列表
    tts_clips = []
    for seg in audio_timing:
        try:
            audio_clip = AudioFileClip(seg['audio_path'])
            # 增加 TTS 音量，确保配音能被清晰听到
            audio_clip = audio_clip.volumex(1.8)
            # 设置音频开始时间
            audio_clip = audio_clip.set_start(seg['actual_start'])
            # 确保不超过视频时长
            if seg['actual_start'] < video_duration:
                tts_clips.append(audio_clip)
        except Exception as e:
            print(f"加载音频失败 {seg['audio_path']}: {str(e)}")
            continue
    
    if not tts_clips:
        print("没有有效的 TTS 音频片段")
        video.close()
        return None
    
    # 合成音频
    if keep_original_audio and video.audio is not None:
        # 降低原声音量，让 TTS 配音更突出
        original_audio = video.audio.volumex(0.4)
        all_audio_clips = [original_audio] + tts_clips
        final_audio = CompositeAudioClip(all_audio_clips)
    else:
        # 只使用 TTS 音频
        final_audio = CompositeAudioClip(tts_clips)
    
    # 设置视频音频
    final_video = video.set_audio(final_audio)
    
    # 导出视频
    final_video.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp_audio_file.m4a',
        remove_temp=True,
        verbose=False,
        logger=None
    )
    
    # 清理资源
    for clip in tts_clips:
        clip.close()
    video.close()
    
    return output_path


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """上传视频文件"""
    if 'video' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件格式'}), 400
    
    # 生成唯一文件名
    file_id = str(uuid.uuid4())
    ext = file.filename.rsplit('.', 1)[1].lower()
    filename = f"{file_id}.{ext}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    file.save(filepath)
    
    # 获取视频信息
    cap = cv2.VideoCapture(filepath)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    return jsonify({
        'file_id': file_id,
        'filename': filename,
        'duration': duration,
        'video_url': url_for('serve_video', filename=filename)
    })

@app.route('/analyze', methods=['POST'])
def analyze_video():
    """分析视频并生成字幕"""
    data = request.json
    file_id = data.get('file_id')
    interval = data.get('interval', 2)
    detection_mode = data.get('detection_mode', 'interval')  # 'interval' 或 'content'
    threshold = data.get('threshold', 0.15)  # 内容变化阈值
    language = data.get('language', 'zh')  # 字幕语言 'zh' 或 'en'
    
    if not file_id:
        return jsonify({'error': '缺少文件ID'}), 400
    
    # 查找视频文件
    video_path = None
    for ext in ALLOWED_EXTENSIONS:
        path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.{ext}")
        if os.path.exists(path):
            video_path = path
            break
    
    if not video_path:
        return jsonify({'error': '视频文件不存在'}), 404
    
    try:
        subtitles, duration = analyze_video_events(
            video_path, 
            interval_seconds=interval,
            detection_mode=detection_mode,
            threshold=threshold,
            language=language
        )
        
        # 保存CSV字幕文件
        csv_content = generate_csv(subtitles)
        csv_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}.csv")
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            f.write(csv_content)
        
        return jsonify({
            'subtitles': subtitles,
            'duration': duration,
            'csv_url': url_for('download_csv', file_id=file_id)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/burn_subtitles', methods=['POST'])
def burn_subtitles_route():
    """将字幕烧录到视频"""
    data = request.json
    file_id = data.get('file_id')
    subtitles = data.get('subtitles', [])
    
    if not file_id or not subtitles:
        return jsonify({'error': '缺少必要参数'}), 400
    
    # 查找视频文件
    video_path = None
    for ext in ALLOWED_EXTENSIONS:
        path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.{ext}")
        if os.path.exists(path):
            video_path = path
            break
    
    if not video_path:
        return jsonify({'error': '视频文件不存在'}), 404
    
    try:
        output_filename = f"{file_id}_subtitled.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        burn_subtitles(video_path, subtitles, output_path)
        
        return jsonify({
            'video_url': url_for('serve_output', filename=output_filename)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/videos/<filename>')
def serve_video(filename):
    """提供上传的视频文件"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/output/<filename>')
def serve_output(filename):
    """提供输出的视频文件"""
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))

@app.route('/download_csv/<file_id>')
def download_csv(file_id):
    """下载CSV字幕文件"""
    csv_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}.csv")
    if os.path.exists(csv_path):
        return send_file(csv_path, as_attachment=True, download_name='subtitles.csv')
    return jsonify({'error': '字幕文件不存在'}), 404


@app.route('/generate_voiceover', methods=['POST'])
def generate_voiceover():
    """生成AI配音视频"""
    data = request.json
    file_id = data.get('file_id')
    subtitles = data.get('subtitles', [])
    language = data.get('language', 'zh')
    voice_gender = data.get('voice_gender', 'female')
    keep_original_audio = data.get('keep_original_audio', True)
    
    if not file_id or not subtitles:
        return jsonify({'error': '缺少必要参数'}), 400
    
    # 查找视频文件
    video_path = None
    for ext in ALLOWED_EXTENSIONS:
        path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.{ext}")
        if os.path.exists(path):
            video_path = path
            break
    
    if not video_path:
        return jsonify({'error': '视频文件不存在'}), 404
    
    try:
        # 创建临时音频目录
        temp_audio_dir = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_tts")
        os.makedirs(temp_audio_dir, exist_ok=True)
        
        # 生成 TTS 音频
        audio_segments = generate_tts_for_subtitles(
            subtitles, 
            language=language, 
            voice_gender=voice_gender,
            output_dir=temp_audio_dir
        )
        
        if not audio_segments:
            return jsonify({'error': 'TTS 生成失败，没有生成任何音频'}), 500
        
        # 计算音频时间轴（处理重叠）
        audio_timing = calculate_audio_timing(audio_segments)
        
        # 合成视频
        output_filename = f"{file_id}_voiceover.mp4"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        result = merge_audio_with_video(
            video_path, 
            audio_timing, 
            output_path, 
            keep_original_audio=keep_original_audio
        )
        
        if not result:
            return jsonify({'error': '视频合成失败'}), 500
        
        # 清理临时音频文件
        import shutil
        try:
            shutil.rmtree(temp_audio_dir)
        except:
            pass
        
        return jsonify({
            'message': f'成功为 {len(audio_timing)} 条字幕生成配音',
            'video_url': url_for('serve_output', filename=output_filename),
            'timing_info': [{
                'id': t['id'],
                'original_start': round(t['original_start'], 2),
                'actual_start': round(t['actual_start'], 2),
                'duration': round(t['duration'], 2),
                'adjusted': t['adjusted']
            } for t in audio_timing]
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7860)
