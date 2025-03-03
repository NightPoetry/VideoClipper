import sys  
import os  
import cv2  
import numpy as np  
import subprocess  
from moviepy.editor import VideoFileClip  
import tempfile  
import threading  
import time  
import shutil  
# 在 video_clipper.py 顶部添加  
import numpy.core._multiarray_umath  
numpy.core._multiarray_umath.__file__  # 触发实际加载
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, QSlider,   
                            QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QListWidget,   
                            QListWidgetItem, QMessageBox, QProgressBar, QCheckBox, QComboBox)  
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon  
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QUrl, QEvent  
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent  

# 添加PyOpenCL支持 - 直接使用GPU而不依赖OpenCV的OpenCL实现  
try:  
    import pyopencl as cl  
    PYOPENCL_AVAILABLE = True  
except ImportError:  
    PYOPENCL_AVAILABLE = False  

# OpenCL 加速器类  
class OpenCLAccelerator:  
    """使用PyOpenCL实现的独立GPU加速器"""  
    
    def __init__(self):  
        self.initialized = False  
        self.ctx = None  
        self.queue = None  
        self.program = None  
        self.device_name = "未知设备"  
        
        # 定义OpenCL内核  
        self.kernel_code = """  
        __kernel void process_frame(  
            __global const uchar4* input,  
            __global uchar4* output,  
            int width,  
            int height)  
        {  
            // 获取全局ID  
            int gid_x = get_global_id(0);  
            int gid_y = get_global_id(1);  
            
            // 检查边界  
            if (gid_x >= width || gid_y >= height)  
                return;  
                
            // 计算位置  
            int idx = gid_y * width + gid_x;  
            
            // 简单处理：拷贝像素值  
            // 在这里可以添加更复杂的图像处理操作  
            output[idx] = input[idx];  
        }  
        """  
        
        # 尝试初始化OpenCL  
        self.initialize()  
        
    def initialize(self):  
        """初始化OpenCL上下文、队列和程序"""  
        if not PYOPENCL_AVAILABLE:  
            print("PyOpenCL不可用，无法使用GPU加速")  
            return False  
            
        try:  
            # 获取平台和设备  
            platforms = cl.get_platforms()  
            if not platforms:  
                print("未找到OpenCL平台")  
                return False  
                
            # 尝试找到GPU设备  
            gpu_devices = []  
            for platform in platforms:  
                try:  
                    platform_devices = platform.get_devices(device_type=cl.device_type.GPU)  
                    gpu_devices.extend(platform_devices)  
                except:  
                    pass  
            
            # 如果没有GPU设备，尝试找CPU设备  
            if not gpu_devices:  
                for platform in platforms:  
                    try:  
                        platform_devices = platform.get_devices(device_type=cl.device_type.CPU)  
                        gpu_devices.extend(platform_devices)  
                    except:  
                        pass  
                        
            if not gpu_devices:  
                print("未找到支持OpenCL的设备")  
                return False  
                
            # 使用找到的第一个设备  
            device = gpu_devices[0]  
            self.device_name = device.name  
            
            # 创建上下文和命令队列  
            self.ctx = cl.Context([device])  
            self.queue = cl.CommandQueue(self.ctx)  
            
            # 编译程序  
            self.program = cl.Program(self.ctx, self.kernel_code).build()  
            
            self.initialized = True  
            print(f"OpenCL初始化成功，使用设备: {self.device_name}")  
            return True  
            
        except Exception as e:  
            print(f"OpenCL初始化失败: {str(e)}")  
            return False  
    
    def process_frame(self, frame):  
        """使用OpenCL处理帧"""  
        if not self.initialized:  
            return frame  
            
        try:  
            # 确保帧是RGBA格式  
            if frame.shape[2] == 3:  # BGR/RGB  
                frame_rgba = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)  
                frame_rgba[:, :, 0:3] = frame  
                frame_rgba[:, :, 3] = 255  # Alpha通道  
            else:  
                frame_rgba = frame  
                
            height, width = frame_rgba.shape[:2]  
            
            # 创建输入和输出缓冲区  
            frame_flat = frame_rgba.reshape(-1, 4).astype(np.uint8)  
            input_buf = cl.Buffer(self.ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=frame_flat)  
            output_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, frame_flat.nbytes)  
            
            # 执行内核  
            global_size = (width, height)  
            local_size = None  # 让OpenCL自动选择本地工作组大小  
            
            self.program.process_frame(self.queue, global_size, local_size,   
                                      input_buf, output_buf, np.int32(width), np.int32(height))  
            
            # 读回结果  
            result = np.empty_like(frame_flat)  
            cl.enqueue_copy(self.queue, result, output_buf)  
            
            # 重塑回原始形状  
            processed_frame = result.reshape(height, width, 4)  
            
            # 如果原始帧是3通道，转回3通道  
            if frame.shape[2] == 3:  
                return processed_frame[:, :, 0:3]  
            else:  
                return processed_frame  
                
        except Exception as e:  
            print(f"OpenCL处理帧失败: {str(e)}")  
            return frame  

# GPU加速检测  
def check_gpu_acceleration():  
    acceleration_info = {  
        'openCL': False,  
        'cuda': False,  
        'method': 'cpu',  # 默认使用CPU  
        'device_name': 'CPU'  
    }  
    
    # 创建OpenCL加速器  
    opencl_accelerator = OpenCLAccelerator()  
    if opencl_accelerator.initialized:  
        acceleration_info['openCL'] = True  
        acceleration_info['method'] = 'openCL'  
        acceleration_info['device_name'] = opencl_accelerator.device_name  
        acceleration_info['accelerator'] = opencl_accelerator  
        return acceleration_info  
    
    # 如果OpenCL初始化失败，检查OpenCV的CUDA支持  
    try:  
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()  
        if cuda_devices > 0:  
            acceleration_info['cuda'] = True  
            acceleration_info['method'] = 'cuda'  
            acceleration_info['device_name'] = f"CUDA设备 ({cuda_devices}个)"  
            print(f"CUDA加速可用: 发现{cuda_devices}个设备")  
        else:  
            print("未检测到CUDA设备")  
    except:  
        print("OpenCV没有编译CUDA支持")  
    
    # 打印最终使用的加速方法  
    print(f"将使用 {acceleration_info['method'].upper()} 处理视频，设备: {acceleration_info['device_name']}")  
    return acceleration_info  

# 使用GPU加速处理视频帧  
def process_frame_with_acceleration(frame, acceleration_info):  
    method = acceleration_info['method']  
    
    if method == 'openCL' and 'accelerator' in acceleration_info:  
        # 使用嵌入式OpenCL加速器  
        return acceleration_info['accelerator'].process_frame(frame)  
            
    elif method == 'cuda':  
        try:  
            # 将帧上传到GPU  
            gpu_frame = cv2.cuda_GpuMat()  
            gpu_frame.upload(frame)  
            
            # 这里可以添加CUDA加速的处理操作  
            # 例如: gpu_frame = cv2.cuda.resize(gpu_frame, (frame.shape[1], frame.shape[0]))  
            
            # 将处理后的帧下载回CPU  
            result = gpu_frame.download()  
            return result  
        except Exception as e:  
            print(f"CUDA处理失败: {str(e)}")  
            return frame  
    
    # 默认返回原始帧  
    return frame  

class AudioPreview:  
    def __init__(self):  
        self.media_player = QMediaPlayer()  
        self.audio_playing = False  
        self.temp_audio_file = None  
        self.volume = 50  # 默认音量50%  
        self.media_player.setVolume(self.volume)  
        self.current_audio_position = 0  
        
    def play_audio_from_clip(self, video_clip, start_time, duration=5.0):  
        """从视频片段中提取并播放音频片段"""  
        try:  
            # 停止之前的音频播放  
            self.stop_audio()  
            
            # 记录音频起始位置  
            self.current_audio_position = start_time  
            
            # 从MoviePy的视频片段提取音频  
            if video_clip.audio is not None:  
                # 提取指定时间范围的音频  
                end_time = min(start_time + duration, video_clip.duration)  
                audio_segment = video_clip.audio.subclip(start_time, end_time)  
                
                # 创建临时目录(如果不存在)  
                temp_dir = os.path.join(tempfile.gettempdir(), "videoslicer")  
                os.makedirs(temp_dir, exist_ok=True)  
                
                # 导出临时音频文件  
                self.temp_audio_file = os.path.join(temp_dir, f"temp_audio_{int(start_time*1000)}.wav")  
                audio_segment.write_audiofile(self.temp_audio_file, logger=None)  
                
                # 使用QMediaPlayer播放  
                self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(self.temp_audio_file)))  
                self.media_player.play()  
                self.audio_playing = True  
            else:  
                print("视频没有音轨")  
        except Exception as e:  
            print(f"音频预览错误: {str(e)}")  
    
    def stop_audio(self):  
        """停止当前音频播放"""  
        if self.audio_playing:  
            self.media_player.stop()  
            self.audio_playing = False  
            # 清理临时文件  
            self.cleanup_temp_files()  
    
    def cleanup_temp_files(self):  
        """清理临时音频文件"""  
        if self.temp_audio_file and os.path.exists(self.temp_audio_file):  
            try:  
                os.remove(self.temp_audio_file)  
                self.temp_audio_file = None  
            except:  
                pass  # 如果无法删除，忽略错误  
            
    def set_volume(self, volume):  
        """设置音量 (0-100)"""  
        self.volume = volume  
        self.media_player.setVolume(volume)  

class VolumeControl(QWidget):  
    """音量控制组件"""  
    def __init__(self, parent=None):  
        super().__init__(parent)  
        self.setup_ui()  
        
    def setup_ui(self):  
        layout = QHBoxLayout(self)  
        layout.setContentsMargins(0, 0, 0, 0)  
        
        # 音量标签  
        self.label = QLabel("音量:")  
        layout.addWidget(self.label)  
        
        # 音量滑块  
        self.slider = QSlider(Qt.Horizontal)  
        self.slider.setMinimum(0)  
        self.slider.setMaximum(100)  
        self.slider.setValue(50)  # 默认音量50%  
        self.slider.setFixedWidth(100)  
        layout.addWidget(self.slider)  
        
        # 音量值显示  
        self.value_label = QLabel("50%")  
        self.value_label.setFixedWidth(40)  
        layout.addWidget(self.value_label)  
        
        # 连接信号  
        self.slider.valueChanged.connect(self.on_volume_changed)  
        
    def on_volume_changed(self, value):  
        """音量改变时更新显示"""  
        self.value_label.setText(f"{value}%")  
        
    def get_volume(self):  
        """获取当前音量值"""  
        return self.slider.value()  
        
    def set_volume(self, volume):  
        """设置音量滑块值"""  
        self.slider.setValue(volume)  

class SpeedControlWidget(QWidget):  
    speedChanged = pyqtSignal(float)  
    
    def __init__(self, parent=None):  
        super().__init__(parent)  
        self.speeds = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]  
        self.current_speed_index = 2  # 默认1.0倍速  
        
        # 创建UI  
        self.layout = QHBoxLayout(self)  
        self.layout.setContentsMargins(0, 0, 0, 0)  
        
        self.speed_label = QLabel("播放速度:")  
        self.layout.addWidget(self.speed_label)  
        
        self.slower_btn = QPushButton("<<")  
        self.slower_btn.clicked.connect(self.decrease_speed)  
        self.layout.addWidget(self.slower_btn)  
        
        self.speed_value = QLabel(f"{self.speeds[self.current_speed_index]}x")  
        self.speed_value.setMinimumWidth(40)  
        self.speed_value.setAlignment(Qt.AlignCenter)  
        self.layout.addWidget(self.speed_value)  
        
        self.faster_btn = QPushButton(">>")  
        self.faster_btn.clicked.connect(self.increase_speed)  
        self.layout.addWidget(self.faster_btn)  
    
    def decrease_speed(self):  
        if self.current_speed_index > 0:  
            self.current_speed_index -= 1  
            self.update_speed()  
    
    def increase_speed(self):  
        if self.current_speed_index < len(self.speeds) - 1:  
            self.current_speed_index += 1  
            self.update_speed()  
    
    def update_speed(self):  
        current_speed = self.speeds[self.current_speed_index]  
        self.speed_value.setText(f"{current_speed}x")  
        self.speedChanged.emit(current_speed)  
        
    def get_current_speed(self):  
        return self.speeds[self.current_speed_index]  

class ClipListItem:  
    def __init__(self, start_frame, end_frame, fps, index):  
        self.start_frame = start_frame  
        self.end_frame = end_frame  
        self.fps = fps  
        self.index = index  
        self.start_time = start_frame / fps  
        self.end_time = end_frame / fps  
        self.duration = self.end_time - self.start_time  

    def get_info_text(self):  
        return f"片段 {self.index}: {self.start_time:.2f}s - {self.end_time:.2f}s (时长: {self.duration:.2f}s)"  

class VideoClipperApp(QMainWindow):  
    def __init__(self):  
        super().__init__()  
        self.setWindowTitle("视频切片器")  
        self.setGeometry(100, 100, 1200, 800)  
        
        # 安装事件过滤器以捕获全局快捷键  
        self.installEventFilter(self)  
        
        # 初始化视频相关变量  
        self.video_clip = None  
        self.current_frame = 0  
        self.total_frames = 0  
        self.fps = 0  
        self.duration = 0  
        self.is_playing = False  
        self.playback_speed = 1.0  
        self.start_frame = None  
        self.end_frame = None  
        self.clip_list = []  
        self.clip_mode = "start"  # 切片模式：start或end  
        
        # 检测GPU加速选项  
        self.acceleration_info = check_gpu_acceleration()  
        
        # 初始化播放计时器  
        self.playback_timer = QTimer(self)  
        self.playback_timer.timeout.connect(self.update_frame)  
        
        # 添加音频预览组件  
        self.audio_preview = AudioPreview()  
        
        # 创建主布局  
        central_widget = QWidget()  
        self.setCentralWidget(central_widget)  
        main_layout = QHBoxLayout(central_widget)  
        
        # 创建左侧布局（视频显示和控制）  
        left_panel = QWidget()  
        left_layout = QVBoxLayout(left_panel)  
        
        # 添加快捷键提示标签在顶部  
        shortcuts_label = QLabel("快捷键: 回车=播放/暂停 | 空格=设置起点/终点(按两次空格自动添加片段)")  
        shortcuts_label.setAlignment(Qt.AlignCenter)  
        shortcuts_label.setStyleSheet("color: #333333; background-color: #f0f0f0; padding: 5px; border-radius: 3px;")  
        left_layout.addWidget(shortcuts_label)  
        
        # 视频显示区域  
        self.video_display = QLabel("加载视频后将在此处显示")  
        self.video_display.setAlignment(Qt.AlignCenter)  
        self.video_display.setStyleSheet("background-color: black; color: white;")  
        self.video_display.setMinimumSize(640, 360)  
        left_layout.addWidget(self.video_display, 3)  
        
        # 视频信息显示  
        self.info_label = QLabel("视频信息: 尚未加载")  
        left_layout.addWidget(self.info_label)  
        
        # 时间显示  
        info_widget = QWidget()  
        info_layout = QHBoxLayout(info_widget)  
        info_layout.setContentsMargins(0, 0, 0, 0)  
        
        self.time_label = QLabel("时间: 0.00s / 0.00s")  
        info_layout.addWidget(self.time_label)  
        
        self.clip_info_label = QLabel("")  
        self.clip_info_label.setStyleSheet("color: green;")  
        info_layout.addWidget(self.clip_info_label, 1)  
        
        # 添加音量控制  
        self.volume_control = VolumeControl()  
        self.volume_control.slider.valueChanged.connect(self.on_volume_changed)  
        info_layout.addWidget(self.volume_control)  
        
        left_layout.addWidget(info_widget)  
        
        # 帧滑块  
        self.frame_slider = QSlider(Qt.Horizontal)  
        self.frame_slider.setMinimum(0)  
        self.frame_slider.setMaximum(100)  # 将在加载视频后更新  
        self.frame_slider.valueChanged.connect(self.on_slider_change)  
        left_layout.addWidget(self.frame_slider)  
        
        # 控制按钮  
        controls_widget = QWidget()  
        self.controls_layout = QHBoxLayout(controls_widget)  
        self.controls_layout.setContentsMargins(0, 0, 0, 0)  
        
        self.load_btn = QPushButton("加载视频")  
        self.load_btn.clicked.connect(self.load_video)  
        self.controls_layout.addWidget(self.load_btn)  
        
        self.play_btn = QPushButton("播放")  
        self.play_btn.clicked.connect(self.toggle_playback)  
        self.controls_layout.addWidget(self.play_btn)  
        
        self.set_start_btn = QPushButton("设置起点")  
        self.set_start_btn.clicked.connect(self.set_start_point)  
        self.controls_layout.addWidget(self.set_start_btn)  
        
        self.set_end_btn = QPushButton("设置终点")  
        self.set_end_btn.clicked.connect(self.set_end_point)  
        self.controls_layout.addWidget(self.set_end_btn)  
        
        self.add_clip_btn = QPushButton("添加片段")  
        self.add_clip_btn.clicked.connect(self.add_clip)  
        self.controls_layout.addWidget(self.add_clip_btn)  
        
        left_layout.addWidget(controls_widget)  
        
        # 添加速度控制  
        self.speed_control = SpeedControlWidget()  
        self.speed_control.speedChanged.connect(self.on_speed_changed)  
        left_layout.addWidget(self.speed_control)  
        
        # 片段导出控制  
        export_widget = QWidget()  
        self.export_layout = QVBoxLayout(export_widget)  
        
        export_controls = QWidget()  
        export_controls_layout = QHBoxLayout(export_controls)  
        export_controls_layout.setContentsMargins(0, 0, 0, 0)  
        
        self.export_btn = QPushButton("导出所选片段")  
        self.export_btn.clicked.connect(self.export_selected_clip)  
        export_controls_layout.addWidget(self.export_btn)  
        
        self.export_all_btn = QPushButton("导出所有片段")  
        self.export_all_btn.clicked.connect(self.export_all_clips)  
        export_controls_layout.addWidget(self.export_all_btn)  
        
        self.include_audio_checkbox = QCheckBox("包含音频")  
        self.include_audio_checkbox.setChecked(True)  
        export_controls_layout.addWidget(self.include_audio_checkbox)  
        
        # 添加GPU加速选项  
        self.use_gpu_checkbox = QCheckBox("导出时使用GPU加速")  
        self.use_gpu_checkbox.setChecked(True)  
        export_controls_layout.addWidget(self.use_gpu_checkbox)  
        
        self.export_layout.addWidget(export_controls)  
        
        # 导出进度条  
        self.progress_bar = QProgressBar()  
        self.progress_bar.setVisible(False)  
        self.export_layout.addWidget(self.progress_bar)  
        
        left_layout.addWidget(export_widget)  
        
        # 添加加速状态显示  
        self.acceleration_label = QLabel(f"加速模式: {self.acceleration_info['method'].upper()} (设备: {self.acceleration_info['device_name']})")  
        self.acceleration_label.setAlignment(Qt.AlignCenter)  
        self.acceleration_label.setStyleSheet("color: blue; font-weight: bold;")  
        left_layout.addWidget(self.acceleration_label)  
        
        # 添加左侧面板到主布局  
        main_layout.addWidget(left_panel, 7)  
        
        # 创建右侧布局（片段列表）  
        right_panel = QWidget()  
        right_layout = QVBoxLayout(right_panel)  
        
        # 片段列表标题  
        clips_title = QLabel("已添加的片段")  
        clips_title.setFont(QFont("Arial", 12, QFont.Bold))  
        right_layout.addWidget(clips_title)  
        
        # 片段列表  
        self.clip_list_widget = QListWidget()  
        self.clip_list_widget.itemDoubleClicked.connect(self.preview_clip)  
        right_layout.addWidget(self.clip_list_widget, 1)  
        
        # 删除片段按钮  
        self.delete_clip_btn = QPushButton("删除所选片段")  
        self.delete_clip_btn.clicked.connect(self.delete_selected_clip)  
        right_layout.addWidget(self.delete_clip_btn)  
        
        # 添加右侧面板到主布局  
        main_layout.addWidget(right_panel, 3)  
        
        # 设置焦点策略，确保空格键和回车键可以被全局捕获  
        self.setup_focus_policy()  
        
        # 设置初始状态  
        self.update_ui_state(False)  
    
    def setup_focus_policy(self):  
        """设置所有按钮的焦点策略，防止它们捕获空格键和回车键"""  
        buttons = [  
            self.load_btn, self.play_btn, self.set_start_btn, self.set_end_btn,  
            self.add_clip_btn, self.export_btn, self.export_all_btn, self.delete_clip_btn,  
            self.speed_control.slower_btn, self.speed_control.faster_btn  
        ]  
        
        for button in buttons:  
            # 设置为NoFocus，使按钮不接收键盘焦点  
            button.setFocusPolicy(Qt.NoFocus)  
        
        # 设置滑块只在点击时获得焦点  
        self.frame_slider.setFocusPolicy(Qt.ClickFocus)  
        self.volume_control.slider.setFocusPolicy(Qt.ClickFocus)  
        
        # 让主窗口默认获得焦点  
        self.setFocusPolicy(Qt.StrongFocus)  
    
    def eventFilter(self, obj, event):  
        """事件过滤器，处理全局键盘事件"""  
        if event.type() == QEvent.KeyPress:  # 使用 QEvent.KeyPress 而不是 Qt.KeyPress  
            # 只有在视频已加载时才处理快捷键  
            if self.video_clip is not None:  
                if event.key() == Qt.Key_Space:  
                    # 处理空格键逻辑，无论焦点在哪里  
                    if self.clip_mode == "start":  
                        # 设置起点  
                        self.set_start_point()  
                        self.clip_mode = "end"  
                    else:  
                        # 设置终点  
                        self.set_end_point()  
                        
                        # 自动添加当前片段到列表  
                        self.add_clip()  
                        
                        # 重置为起点模式，准备下一个片段  
                        self.clip_mode = "start"  
                    
                    # 事件已处理  
                    return True  
                
                elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:  
                    # 处理回车键逻辑：播放/暂停  
                    self.toggle_playback()  
                    return True  
        
        # 其他事件交给默认处理程序  
        return super().eventFilter(obj, event)  
    
    def update_ui_state(self, video_loaded):  
        """更新UI状态"""  
        self.play_btn.setEnabled(video_loaded)  
        self.set_start_btn.setEnabled(video_loaded)  
        self.set_end_btn.setEnabled(video_loaded)  
        self.add_clip_btn.setEnabled(video_loaded and self.start_frame is not None and self.end_frame is not None)  
        self.export_btn.setEnabled(len(self.clip_list) > 0)  
        self.export_all_btn.setEnabled(len(self.clip_list) > 0)  
        self.delete_clip_btn.setEnabled(len(self.clip_list) > 0)  
        
        if video_loaded:  
            self.play_btn.setText("播放" if not self.is_playing else "暂停")  
    
    def load_video(self):  
        """加载视频文件"""  
        file_path, _ = QFileDialog.getOpenFileName(  
            self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mkv *.mov);;所有文件 (*)"  
        )  
        
        if file_path:  
            try:  
                # 停止之前的播放  
                if self.is_playing:  
                    self.stop_playback()  
                
                # 清除之前的片段  
                self.clip_list.clear()  
                self.clip_list_widget.clear()  
                self.start_frame = None  
                self.end_frame = None  
                self.clip_info_label.setText("")  
                
                # 重置剪辑模式  
                self.clip_mode = "start"  
                
                # 加载视频  
                self.video_clip = VideoFileClip(file_path)  
                self.fps = self.video_clip.fps  
                self.duration = self.video_clip.duration  
                self.total_frames = int(self.fps * self.duration)  
                self.current_frame = 0  
                
                # 更新视频信息  
                self.info_label.setText(f"视频信息: {os.path.basename(file_path)} | {self.video_clip.size[0]}x{self.video_clip.size[1]} | {self.fps:.2f}fps | {self.duration:.2f}秒")  
                
                # 更新进度条  
                self.frame_slider.setMaximum(self.total_frames - 1)  
                self.frame_slider.setValue(0)  
                
                # 显示第一帧  
                self.update_frame()  
                
                # 更新UI状态  
                self.update_ui_state(True)  
                
                # 显示加载成功  
                QMessageBox.information(self, "成功", "视频加载成功!")  
                
            except Exception as e:  
                QMessageBox.critical(self, "错误", f"加载视频时出错: {str(e)}")  
    
    def on_volume_changed(self, volume):  
        """处理音量变化"""  
        self.audio_preview.set_volume(volume)  
    
    def update_frame(self):  
        """更新视频帧显示"""  
        if self.video_clip is not None and self.current_frame < self.total_frames:  
            try:  
                # 获取当前帧  
                time_pos = self.current_frame / self.fps  
                frame = self.video_clip.get_frame(time_pos)  
                
                # 预览时直接使用CPU处理，不使用GPU加速  
                # 直接使用原始帧进行显示  
                
                # 转换为RGB格式  
                if frame.shape[2] == 4:  # 如果是RGBA  
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)  
                
                # 显示帧  
                height, width, channel = frame.shape  
                bytes_per_line = 3 * width  
                q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)  
                pixmap = QPixmap.fromImage(q_img)  
                
                # 根据窗口大小调整图像  
                display_size = self.video_display.size()  
                scaled_pixmap = pixmap.scaled(display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)  
                self.video_display.setPixmap(scaled_pixmap)  
                
                # 更新时间显示  
                self.time_label.setText(f"时间: {time_pos:.2f}s / {self.duration:.2f}s")  
                
                # 如果是播放模式，更新帧  
                if self.is_playing:  
                    # 根据速度调整帧更新步长  
                    step = max(1, int(self.playback_speed))  
                    self.current_frame += step  
                    
                    # 检查是否到达视频末尾  
                    if self.current_frame >= self.total_frames:  
                        self.current_frame = self.total_frames - 1  
                        self.stop_playback()  
                    
                    # 更新滑块位置（禁用valueChanged信号以避免递归）  
                    self.frame_slider.blockSignals(True)  
                    self.frame_slider.setValue(self.current_frame)  
                    self.frame_slider.blockSignals(False)  
            except Exception as e:  
                print(f"更新帧错误: {str(e)}")  
    
    def on_slider_change(self, value):  
        """处理滑块位置变化"""  
        if self.video_clip is not None:  
            # 记录当前帧与目标帧的差距，用于判断是否需要重新播放音频  
            previous_frame = self.current_frame  
            self.current_frame = value  
            
            # 更新视频显示  
            self.update_frame()  
            
            # 音频同步逻辑改进 - 判断拖动幅度是否需要重新播放音频  
            # 如果拖动超过0.5秒，或者是在播放中拖动，都重新播放音频  
            if self.is_playing or abs(previous_frame - self.current_frame) > (self.fps / 2):  
                self.audio_preview.stop_audio()  
                current_time = self.current_frame / self.fps  
                if self.playback_speed <= 2.0:  # 只在较低的速度播放音频  
                    self.audio_preview.play_audio_from_clip(self.video_clip, current_time)  
    
    def toggle_playback(self):  
        """切换播放/暂停状态"""  
        if self.video_clip is not None:  
            if self.is_playing:  
                self.stop_playback()  
            else:  
                self.start_playback()  
    
    def start_playback(self):  
        """开始视频播放"""  
        if not self.is_playing and self.video_clip is not None:  
            self.is_playing = True  
            self.play_btn.setText("暂停")  
            
            # 根据速度调整帧更新间隔  
            interval = int(1000 / (self.fps * self.playback_speed))  
            self.playback_timer.start(max(1, interval))  # 至少1ms  
            
            # 开始音频预览  
            if self.playback_speed <= 2.0:  # 只在较低的速度播放音频  
                current_time = self.current_frame / self.fps  
                self.audio_preview.play_audio_from_clip(self.video_clip, current_time)  
    
    def stop_playback(self):  
        """停止视频播放"""  
        if self.is_playing:  
            self.is_playing = False  
            self.play_btn.setText("播放")  
            self.playback_timer.stop()  
            
            # 停止音频预览  
            self.audio_preview.stop_audio()  
    
    def on_speed_changed(self, speed):  
        """处理播放速度变化"""  
        self.playback_speed = speed  
        # 更新定时器间隔以反映速度变化  
        if self.is_playing:  
            self.stop_playback()  
            self.start_playback()  
    
    def set_start_point(self):  
        """设置片段起点"""  
        if self.video_clip is not None:  
            self.start_frame = self.current_frame  
            self.end_frame = None  # 清除终点，准备重新设置  
            self.update_ui_state(True)  
            
            # 更新剪辑信息显示  
            self.update_clip_info()  
            
            # 显示确认  
            time_pos = self.start_frame / self.fps  
            self.clip_info_label.setText(f"起点: {time_pos:.2f}秒")  
    
    def set_end_point(self):  
        """设置片段终点"""  
        if self.video_clip is not None and self.start_frame is not None:  
            self.end_frame = self.current_frame  
            
            # 如果起点在终点之后，交换两点  
            if self.start_frame > self.end_frame:  
                self.start_frame, self.end_frame = self.end_frame, self.start_frame  
            
            self.update_ui_state(True)  
            # 更新剪辑信息显示  
            self.update_clip_info()  
            
    def update_clip_info(self):  
        """更新剪辑信息显示"""  
        if self.start_frame is not None and self.end_frame is not None:  
            start_time = self.start_frame / self.fps  
            end_time = self.end_frame / self.fps  
            duration = end_time - start_time  
            self.clip_info_label.setText(f"选定: {start_time:.2f}s - {end_time:.2f}s (时长: {duration:.2f}s)")  
        elif self.start_frame is not None:  
            start_time = self.start_frame / self.fps  
            self.clip_info_label.setText(f"起点: {start_time:.2f}秒")  
        elif self.end_frame is not None:  
            end_time = self.end_frame / self.fps  
            self.clip_info_label.setText(f"终点: {end_time:.2f}秒")  
        else:  
            self.clip_info_label.setText("")  
    
    def add_clip(self):  
        """添加当前片段到列表"""  
        if self.video_clip is not None and self.start_frame is not None and self.end_frame is not None:  
            # 确保起点在终点之前  
            if self.start_frame > self.end_frame:  
                self.start_frame, self.end_frame = self.end_frame, self.start_frame  
            
            # 创建片段项  
            clip_index = len(self.clip_list) + 1  
            clip_item = ClipListItem(self.start_frame, self.end_frame, self.fps, clip_index)  
            self.clip_list.append(clip_item)  
            
            # 添加到列表控件  
            item_text = clip_item.get_info_text()  
            list_item = QListWidgetItem(item_text)  
            self.clip_list_widget.addItem(list_item)  
            
            # 清除当前选择，为下一个片段做准备  
            self.start_frame = None  
            self.end_frame = None  
            
            # 更新UI状态  
            self.update_ui_state(True)  
            self.update_clip_info()  

    def preview_clip(self, item):  
        """预览所选片段"""  
        index = self.clip_list_widget.row(item)  
        if 0 <= index < len(self.clip_list):  
            clip = self.clip_list[index]  
            
            # 停止当前播放  
            if self.is_playing:  
                self.stop_playback()  
            
            # 设置到片段起点  
            self.current_frame = clip.start_frame  
            self.frame_slider.setValue(self.current_frame)  
            self.update_frame()  
            
            # 开始播放  
            self.start_playback()  
    
    def delete_selected_clip(self):  
        """删除所选片段"""  
        selected_items = self.clip_list_widget.selectedItems()  
        if selected_items:  
            for item in selected_items:  
                index = self.clip_list_widget.row(item)  
                self.clip_list_widget.takeItem(index)  
                self.clip_list.pop(index)  
            
            # 重新编号片段  
            for i, clip in enumerate(self.clip_list):  
                clip.index = i + 1  
                item_text = clip.get_info_text()  
                self.clip_list_widget.item(i).setText(item_text)  
            
            # 更新UI状态  
            self.update_ui_state(True)  
    
    def export_clip(self, clip, output_path, include_audio=True, use_gpu=True):  
        """导出单个视频片段"""  
        try:  
            # 提取指定时间范围的视频片段  
            subclip = self.video_clip.subclip(clip.start_time, clip.end_time)  
            
            # 根据选项决定是否包含音频  
            if not include_audio and subclip.audio is not None:  
                subclip = subclip.without_audio()  
            
            # 根据选项决定是否使用GPU加速  
            if use_gpu and (self.acceleration_info['openCL'] or self.acceleration_info['cuda']):  
                # 使用GPU加速处理  
                print(f"使用 {self.acceleration_info['method']} 加速导出片段 {clip.index}")  
                
                # 注意：当使用GPU加速时，我们需要一个自定义的write_videofile函数  
                # 或者处理subclip中的每一帧然后再写出  
                # 下面是一个简化的示例，实际情况可能更复杂  
                
                # 如果使用自定义帧处理器对每帧应用加速  
                def gpu_process(get_frame, t):  
                    frame = get_frame(t)  
                    # 使用GPU加速处理帧  
                    processed = process_frame_with_acceleration(frame, self.acceleration_info)  
                    return processed  
                
                # 创建使用GPU处理过的子剪辑  
                if hasattr(subclip, 'fl_image'):  
                    # MoviePy 1.0.0以上版本  
                    gpu_subclip = subclip.fl_image(lambda img: process_frame_with_acceleration(img, self.acceleration_info))  
                else:  
                    # 对于不支持fl_image的版本，使用其他方法  
                    # 这里是一个简单的回退方案  
                    gpu_subclip = subclip  
                
                # 导出处理后的视频  
                gpu_subclip.write_videofile(  
                    output_path,  
                    codec='libx264',  
                    audio_codec='aac' if include_audio and gpu_subclip.audio is not None else None,  
                    preset='medium',  
                    threads=4  
                )  
            else:  
                # 使用标准CPU处理  
                print(f"使用CPU导出片段 {clip.index}")  
                subclip.write_videofile(  
                    output_path,  
                    codec='libx264',  
                    audio_codec='aac' if include_audio and subclip.audio is not None else None,  
                    preset='medium',  
                    threads=4  
                )  
            
            return True  
        except Exception as e:  
            print(f"导出错误: {str(e)}")  
            return False  
    
    def export_selected_clip(self):  
        """导出选定的视频片段"""  
        if not self.clip_list:  
            QMessageBox.warning(self, "警告", "没有可导出的片段")  
            return  
        
        selected_items = self.clip_list_widget.selectedItems()  
        if not selected_items:  
            QMessageBox.warning(self, "警告", "请先选择一个片段")  
            return  
            
        index = self.clip_list_widget.row(selected_items[0])  
        if 0 <= index < len(self.clip_list):  
            clip = self.clip_list[index]  
            
            output_path, _ = QFileDialog.getSaveFileName(  
                self, "导出片段", f"片段_{clip.index}.mp4", "视频文件 (*.mp4)"  
            )  
            
            if output_path:  
                include_audio = self.include_audio_checkbox.isChecked()  
                use_gpu = self.use_gpu_checkbox.isChecked()  
                
                # 显示进度条  
                self.progress_bar.setValue(0)  
                self.progress_bar.setVisible(True)  
                QApplication.processEvents()  
                
                # 导出视频  
                success = self.export_clip(clip, output_path, include_audio, use_gpu)  
                
                self.progress_bar.setVisible(False)  
                
                if success:  
                    QMessageBox.information(self, "成功", f"片段 {clip.index} 已导出至 {output_path}")  
                else:  
                    QMessageBox.critical(self, "错误", f"导出片段 {clip.index} 失败")  
    
    def export_all_clips(self):  
        """导出所有视频片段"""  
        if not self.clip_list:  
            QMessageBox.warning(self, "警告", "没有可导出的片段")  
            return  
        
        # 选择导出目录  
        export_dir = QFileDialog.getExistingDirectory(self, "选择导出目录")  
        if not export_dir:  
            return  
        
        include_audio = self.include_audio_checkbox.isChecked()  
        use_gpu = self.use_gpu_checkbox.isChecked()  
        
        # 显示进度条  
        self.progress_bar.setValue(0)  
        self.progress_bar.setMaximum(len(self.clip_list))  
        self.progress_bar.setVisible(True)  
        
        # 导出所有片段  
        success_count = 0  
        for i, clip in enumerate(self.clip_list):  
            output_path = os.path.join(export_dir, f"片段_{clip.index}.mp4")  
            
            # 更新进度条  
            self.progress_bar.setValue(i)  
            QApplication.processEvents()  
            
            # 导出视频  
            if self.export_clip(clip, output_path, include_audio, use_gpu):  
                success_count += 1  
        
        self.progress_bar.setVisible(False)  
        
        if success_count > 0:  
            QMessageBox.information(self, "成功", f"成功导出 {success_count}/{len(self.clip_list)} 个片段至 {export_dir}")  
        else:  
            QMessageBox.critical(self, "错误", "所有片段导出失败")  

    def closeEvent(self, event):  
        """程序关闭时清理资源"""  
        # 停止播放  
        if self.is_playing:  
            self.stop_playback()  
            
        # 停止音频并清理临时文件  
        self.audio_preview.stop_audio()  
        self.audio_preview.cleanup_temp_files()  
            
        # 关闭视频文件  
        if self.video_clip is not None:  
            try:  
                self.video_clip.close()  
            except:  
                pass  
                
        # 清理临时目录  
        try:  
            temp_dir = os.path.join(tempfile.gettempdir(), "videoslicer")  
            if os.path.exists(temp_dir):  
                shutil.rmtree(temp_dir, ignore_errors=True)  
        except:  
            pass  
            
        event.accept()  

if __name__ == "__main__":  
    app = QApplication(sys.argv)  
    app.setStyle("Fusion")  # 使用Fusion样式以获得现代外观  
    
    window = VideoClipperApp()  
    window.show()  
    
    sys.exit(app.exec_())