# -*- coding: utf-8 -*-
"""
Created on Mon May  5 16:48:59 2025

@author: Pulee
"""

import os
import time
import cv2
import serial
import warnings
import threading
import win32com.client
import tensorflow as tf
import tkinter as tk
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageTk
from tkinter import ttk, filedialog, messagebox
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.core import predict
from deeplabcut.utils import auxiliaryfunctions
from MvCameraControl_class import *

def list_hikrobot_cameras():
    deviceList = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
    if ret != 0 or deviceList.nDeviceNum == 0:
        return []
    cameras = []
    for i in range(deviceList.nDeviceNum):
        dev_info = cast(deviceList.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        name = ''.join(chr(c) for c in dev_info.SpecialInfo.stUsb3VInfo.chModelName if c != 0)
        cameras.append((i, name))
    return cameras

def resize_with_padding(image, target_size=(640, 480)):
    """
    Resize image to fit within target_size, keep aspect ratio,
    and pad with black borders to match exact dimensions.
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded

def capture_loop(self):
    frame = MV_FRAME_OUT()
    self.cam.MV_CC_GetImageBuffer(frame, 1000)
    input_info = MV_CC_INPUT_FRAME_INFO()
    input_info.pData = cast(frame.pBufAddr, POINTER(c_ubyte))
    input_info.nDataLen = frame.stFrameInfo.nFrameLen
    self.cam.MV_CC_InputOneFrame(input_info)

###### Load pre-training model ######
shuffle=1
trainingsetindex=0
gputouse='0'
modelprefix=""
config_path = r"D:\Expriment\DLC\ycyy\tube-ycyy-hes-2025-04-28\config.yaml"  # You need change it according to actual condition

if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
    del os.environ['TF_CUDNN_USE_AUTOTUNE']

if gputouse is not None:
    try:
        auxiliaryfunctions.set_visible_devices(gputouse[0])
    except AttributeError:
        print("[Warning] `set_visible_devices` not found in auxiliaryfunctions. Skipping GPU binding.")

tf.compat.v1.reset_default_graph()
start_path = os.getcwd()

try:
    cfg = auxiliaryfunctions.read_config(config_path)
except Exception as e:
    print(f"[Error] Failed to read config.yaml: {e}")

try:
    trainFraction = cfg["TrainingFraction"][trainingsetindex]
    modelfolder = os.path.join(
        cfg["project_path"],
        str(auxiliaryfunctions.get_model_folder(trainFraction, 
                                                shuffle, 
                                                cfg, 
                                                modelprefix=modelprefix
                                                )
            )
    )
    path_test_config = Path(modelfolder) / "test" / "pose_cfg.yaml"
    dlc_cfg = load_config(str(path_test_config))
except Exception as e:
    print(f"[Error] Failed to load model or config: {e}")

try:
    Snapshots = auxiliaryfunctions.get_snapshots_from_folder(train_folder=Path(modelfolder) / "train")
except AttributeError:
    print("[Error] `get_snapshots_from_folder` not found. Please update DeepLabCut.")

snapshotindex = -1 if cfg.get("snapshotindex") == "all" else cfg.get("snapshotindex", -1)
snapshot_name = Snapshots[snapshotindex]
print(f"Using snapshot: {snapshot_name} from {modelfolder}")

dlc_cfg["init_weights"] = os.path.join(modelfolder, "train", snapshot_name)
trainingsiterations = (dlc_cfg["init_weights"].split(os.sep)[-1]).split("-")[-1]
dlc_cfg["batch_size"] = 1

DLCscorer = auxiliaryfunctions.GetScorerName(cfg, 
                                             shuffle, 
                                             trainFraction, 
                                             trainingsiterations=trainingsiterations
                                             )
if isinstance(DLCscorer, tuple):
    DLCscorer = DLCscorer[0] 
skeleton = cfg['skeleton']
bodyparts = cfg['bodyparts']
sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)
pdindex = pd.MultiIndex.from_product([[DLCscorer], 
                                      dlc_cfg['all_joints_names'], 
                                      ['x', 'y', 'likelihood']], 
                                     names=['scorer', 'bodyparts', 'coords']
                                     )

class BarrierTrigger:
    def __init__(self, root):
        self.root = root
        self.root.title("Barrier Trigger")
        
        self.canvas_width = 640   # Fixed width
        self.canvas_height = 480  # Fixed height
        
        self.selected_resolution = None
        self.target_fps = 30.0
        self.best_resolution = None
        self.resolution_label = None
        self.capture = None
        self.recording = False
        self.previewing = False
        self.video_writer = None

        self.selected_camera_index = 0
        self.save_dir = os.getcwd()
        self.file_prefix = tk.StringVar(value="test")
        self.config_path = ""

        self.setup_ui()
        self.arduino_port = "COM10"   # You need change it according the actual condition
        self.arduino = serial.Serial(self.arduino_port, 9600, timeout=1)
        time.sleep(2)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0)

        # === device select ===
        ttk.Label(frm, text="Camera:").grid(row=0, column=0, sticky="W")
        self.camera_list = list_available_cameras()
        self.cam_dict = {name: idx for idx, name in self.camera_list}
        self.cam_names = list(self.cam_dict.keys())
        self.camera_var = tk.StringVar()
        self.camera_combo = ttk.Combobox(frm, values=self.cam_names, textvariable=self.camera_var, state='readonly')
        self.camera_combo.grid(row=0, column=1)
        if self.cam_names:
            self.camera_combo.current(0)
            self.selected_camera_index = self.cam_dict[self.cam_names[0]]
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)
        
        # === resolution select ===
        ttk.Label(frm, text="Resolution:").grid(row=1, column=0, sticky="W")
        self.resolution_combo = ttk.Combobox(frm, state="readonly")
        self.resolution_combo.grid(row=1, column=1)
        if self.cam_names:
            self.on_camera_selected()
        self.resolution_combo.bind("<<ComboboxSelected>>", self.on_resolution_selected)

        # === fps select ===
        ttk.Label(frm, text="Target FPS:").grid(row=2, column=0, sticky="W")
        self.fps_entry = ttk.Entry(frm)
        self.fps_entry.insert(0, "30.0")
        self.fps_entry.grid(row=2, column=1)

        ttk.Button(frm, text="Set Save Directory", command=self.select_directory).grid(row=4, column=0)
        ttk.Label(frm, text="Filename Prefix:").grid(row=3, column=0, sticky="W")
        ttk.Entry(frm, textvariable=self.file_prefix).grid(row=3, column=1)
        ttk.Button(frm, text="Preview", command=self.toggle_preview).grid(row=4, column=1)
        ttk.Button(frm, text="Start Recording", command=self.start_recording).grid(row=5, column=0)
        ttk.Button(frm, text="Stop Recording", command=self.stop_recording).grid(row=5, column=1)

        self.label_status = ttk.Label(frm, text="Ready")
        self.label_status.grid(row=6, column=0, columnspan=2)

        self.canvas = tk.Label(self.root)
        self.canvas.grid(row=0, column=1)
        self.recording_time_label = tk.Label(self.root, text="", font=("Helvetica", 24), fg="red")
        self.recording_time_label.grid(row=1, column=1, sticky="e", padx=10, pady=10)
        self.resolution_label = ttk.Label(frm, text="Resolution: Not selected")
        self.resolution_label.grid(row=7, column=0, columnspan=2)
        
    def get_supported_resolutions(self, camera_index=None):
        """
        Return a list of supported (width, height) resolutions for the camera.
        """
        if camera_index is None:
            camera_index = self.selected_camera_index
    
        common_resolutions = [
            (3840, 2160),  # 4K UHD
            (2560, 1440),  # QHD
            (1920, 1080),  # Full HD
            (1600, 1200),  # UXGA
            (1280, 1024),  # SXGA
            (1280, 720),   # HD
            (1024, 768),   # XGA
            (800, 600),    # SVGA
            (640, 480),    # VGA
            (320, 240),    # QVGA
        ]
    
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera {camera_index}.")
            return []
    
        supported = []
        for width, height in common_resolutions:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if abs(actual_width - width) <= 20 and abs(actual_height - height) <= 20:
                supported.append((actual_width, actual_height))
    
        cap.release()
        return supported
    
    def on_closing(self):
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("Arduino port closed.")
        self.root.destroy()
    
    def on_camera_selected(self, event=None):
        cam_label = self.camera_var.get()
        if cam_label in self.cam_dict:
            self.selected_camera_index = self.cam_dict[cam_label]
            resolutions = self.get_supported_resolutions(self.selected_camera_index)
            res_strings = [f"{w}x{h}" for w, h in resolutions] or ["640x480"]
            self.resolution_combo["values"] = res_strings
            self.resolution_combo.current(0)
            w, h = map(int, res_strings[0].split("x"))
            self.selected_resolution = (w, h)
    
    def on_resolution_selected(self, event=None):
        res_text = self.resolution_combo.get()
        if "x" in res_text:
            w, h = map(int, res_text.split("x"))
            self.selected_resolution = (w, h)
    
        if self.previewing and self.capture:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    def select_directory(self):
        path = filedialog.askdirectory()
        if path:
            self.save_dir = path

    def start_hikrobot_preview(self):
        # Create and open device
        self.cam = MvCamera()
        deviceList = MV_CC_DEVICE_INFO_LIST()
        MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, deviceList)
        dev_info = cast(deviceList.pDeviceInfo[self.selected_camera_index], POINTER(MV_CC_DEVICE_INFO)).contents
        self.cam.MV_CC_CreateHandle(dev_info)
        self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    
        # Start grabbing
        self.cam.MV_CC_StartGrabbing()
        self.previewing = True
        self.preview_loop()
    
    def preview_loop(self):
        if not self.previewing:
            return
        frame = MV_FRAME_OUT()
        memset(byref(frame), 0, sizeof(frame))
        ret = self.cam.MV_CC_GetImageBuffer(frame, 1000)
        if ret == 0 and frame.pBufAddr:
            image = np.ctypeslib.as_array((c_ubyte * frame.stFrameInfo.nFrameLen).from_address(addressof(frame.pBufAddr.contents)))
            img = image.reshape((frame.stFrameInfo.nHeight, frame.stFrameInfo.nWidth, 3))
            self.display_image(img)
            self.cam.MV_CC_FreeImageBuffer(frame)
        self.root.after(20, self.preview_loop)

    def update_image_on_main_thread(self, frame, pose):
        """Update the GUI canvas with DLC-predicted keypoints and skeleton."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
        # Define default color for keypoints and skeleton
        keypoint_color = (0, 255, 0)   # Green dots
        skeleton_color = (0, 255, 255) # Yellow lines
    
        # Draw all keypoints
        joints = {}
        for idx, name in enumerate(bodyparts):
            x, y, likelihood = pose[idx]
            if likelihood > 0.7:  # Only plot high-confidence points
                x = int(x)
                y = int(y)
                joints[name] = (x, y)
                cv2.circle(frame_rgb, (x, y), radius=5, color=keypoint_color, thickness=-1)
    
        # Draw skeleton lines
        for pair in skeleton:
            p1, p2 = pair
            if p1 in joints and p2 in joints:
                cv2.line(frame_rgb, joints[p1], joints[p2], color=skeleton_color, thickness=2)
    
        # Convert for Tkinter display
        resized_frame = cv2.resize(frame_rgb, (self.canvas_width, self.canvas_height))
        img = Image.fromarray(resized_frame)
        imgtk = ImageTk.PhotoImage(img, master=self.root)
        self.canvas.imgtk = imgtk
        self.canvas.config(image=imgtk)
    
    def start_hikrobot_recording(self):
        record_params = MV_CC_RECORD_PARAM()
        memset(byref(record_params), 0, sizeof(record_params))
        record_params.strFilePath = f"{filename}".encode('ascii')
        record_params.enRecordFmtType = MV_FormatType_AVI
        record_params.nWidth = ...  # get from MV_CC_GetIntValue("Width")
        record_params.nHeight = ...
        record_params.enPixelType = ...  # from MV_CC_GetEnumValue("PixelFormat")
        record_params.fFrameRate = ...  # from MV_CC_GetFloatValue("ResultingFrameRate")
        self.cam.MV_CC_StartRecord(record_params)
    
        self.recording = True
        self.capture_loop()

    def update_recording_time(self):
        """
        Update both the small status label and the large recording timer label.
        """
        if self.recording and self.recording_start_time:
            elapsed = time.time() - self.recording_start_time
            mins, secs = divmod(int(elapsed), 60)
            time_text = f"{mins:02}:{secs:02}"
            self.label_status.config(text=f"Recording... {time_text}")
            self.recording_time_label.config(text=time_text)
            self.root.after(1000, self.update_recording_time)
        else:
            self.recording_time_label.config(text="")

    def stop_camera(self):
        self.recording = False
        self.previewing = False
        if hasattr(self, 'cam') and self.cam:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_StopRecord()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()

    
    def find_first_last(self, condition):
        first_index = np.argmax(condition) if np.any(condition) else None
        last_index = len(condition) - 1 - np.argmax(condition[::-1]) if np.any(condition) else None
        return first_index, last_index
    
    def analyze_stream(self, fps = 30):
        ###### Initiation ######
        # target_interval = 1.0 / fps
        counter = 0
        flag = 0
        triggerTimesA = []
        triggerTimesT = []
        allDuration = []
        PredictedData = []
        names = dlc_cfg["all_joints_names"]
        barrier_idx = names.index("barrier")
        mouth_idx = names.index("mouth")
        pbar = tqdm()
        self.mark_ttl_on()
        self.recording_start_time = time.time()  # Start recording timer
        self.frame_counter = 0
        self.update_recording_time()  # Start updating timer display
        self.root.after(0, lambda: self.label_status.config(text="Recording..."))
        FirstStartTime = time.time()
        while self.recording and self.capture.isOpened():
            startTime = time.time()
            # loop_start = time.time()
            ret, frame = self.capture.read()
            if not ret:
                warnings.warn(f"[Warning] Could not read frame #{counter}")
                break
            self.video_writer.write(frame)
            
            # Compute Actual FPS
            self.frame_counter += 1
            if self.frame_counter % 100 == 0:
                elapsed = time.time() - self.recording_start_time
                measured_fps = self.frame_counter / elapsed
                print(f"[Measure] Actual FPS during recording: {measured_fps:.2f}")
            
            # Frame process and predict the position
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if cfg.get("cropping", False):
                frame_rgb = frame_rgb[cfg["y1"]:cfg["y2"], cfg["x1"]:cfg["x2"]]
            pose = predict.getpose(frame_rgb, dlc_cfg, sess, inputs, outputs)
            PredictedData.append(pose.flatten())
            counter += 1
            pbar.update(1)
            
            # Live display update
            frame_copy = frame.copy()  # Copy the original frame
            self.root.after(0, self.update_image_on_main_thread, frame_copy, pose)
            
            # Judge if need trigger
            if flag == 0:
                barrier = PredictedData[counter-1][3*barrier_idx:3*(barrier_idx+1)]
                mouth = PredictedData[counter-1][3*mouth_idx:3*(mouth_idx+1)]
                if 0 < barrier[0] - mouth[0] < 15\
                    and abs(mouth[1] - barrier[1]) < 15\
                        and mouth[2] > 0.99\
                            and barrier[2] > 0.99\
                                and counter >=2\
                                    and PredictedData[counter-2][3*mouth_idx] < mouth[0]:
                                        self.trigger_laser()
                                        endTime = time.time()
                                        triggerTimesA.append(endTime-FirstStartTime)
                                        triggerTimesT.append(counter/30)
                                        duration = endTime - startTime
                                        allDuration.append(duration)
                                        triggerFrame = counter
                                        print(counter)
                                        flag = 1
            if flag == 1 and counter-triggerFrame >= 10:
                valid_following = 0
                for j in range(triggerFrame+1, counter):
                    mouth_follow = PredictedData[j][3*mouth_idx:3*(mouth_idx+1)]
                    if mouth_follow[2] > 0.99 and PredictedData[triggerFrame][3*mouth_idx] - mouth_follow[0] > 10:
                        valid_following += 1
                if valid_following >= 10:
                    flag = 0
                    
            # # Control the precise FPS
            # elapsed = time.time() - loop_start
            # sleep_time = target_interval - elapsed
            # if sleep_time > 0:
            #     time.sleep(sleep_time)
                
        pbar.close()
        self.capture.release()
        self.video_writer.release()
        if len(PredictedData) == 0:
            print("No frames were analyzed.")
            return None
        
        ###### Data restore ######
        PredictedData1 = np.array(PredictedData)
        df1 = pd.DataFrame(PredictedData1, columns=pdindex)
        output_path_dict1 = os.path.join(self.save_dir, f"{self.file_prefix.get()}_predicted_data.csv")
        df1.to_csv(output_path_dict1, index=False)
        print(f"Results of PredictedData saved to {output_path_dict1}")
        df2 = pd.DataFrame({'ActualTriggerTime': triggerTimesA,'TheoreticalTriggerTime': triggerTimesT, 'AllDuration': allDuration})
        output_path_dict2 = os.path.join(self.save_dir, f"{self.file_prefix.get()}_TriggerLog.csv")
        df2.to_csv(output_path_dict2, index=False)
        self.data_analysis(df1)
        return df1
    
    def data_analysis(self, df):
        """Analyze the predicted data and calculate behavioral parameters."""
        if df.empty:
            print("[Error] No pose data to analyze.")
            return
        
        # === 1. Extract useful columns automatically from config ===
        scorer = df.columns.levels[0][0]
        get = lambda part, axis: df.loc[:, (scorer, part, axis)]
        
        data = {p: {"x": get(p, 'x'), "y": get(p, 'y'), "likelihood": get(p, 'likelihood')} for p in bodyparts}
        
        # === 2. Reliability filter
        thresh = 0.99
        reliable_idx = {
            p: data[p]["likelihood"][data[p]["likelihood"] > thresh].index
            for p in bodyparts
        }

        # Simple check
        for key in ['spacer1_1', 'spacer1_2', 'spacer2_1', 'spacer2_2']:
            if reliable_idx[key].empty:
                print(f"[Warning] {key} not reliably detected. Skipping analysis.")
                return
        
        # === 3. Calculate alignment lines ===
        def get_last_valid_x(part):
            return data[part]["x"].iloc[reliable_idx[part][-1]]
        
        def get_last_valid_y(part):
            return data[part]["y"].iloc[reliable_idx[part][-1]]
    
        try:
            dx1 = get_last_valid_x('spacer1_2') - get_last_valid_x('spacer1_1')
            dy1 = get_last_valid_y('spacer1_2') - get_last_valid_y('spacer1_1')
            k1 = dy1 / dx1
            b1 = get_last_valid_y('spacer1_2') - k1 * get_last_valid_x('spacer1_2')
            dx2 = get_last_valid_x('spacer2_2') - get_last_valid_x('spacer2_1')
            dy2 = get_last_valid_y('spacer2_2') - get_last_valid_y('spacer2_1')
            k2 = dy2 / dx2
            b2 = get_last_valid_y('spacer2_2') - k2 * get_last_valid_x('spacer2_2')
            dx3 = get_last_valid_x('tube2') - get_last_valid_x('tube1')
        except Exception as e:
            print(f"[Error] Failed calculating alignment lines: {e}")
            return
    
        # === 4. Search movement cycles ===
        # Only use neck points for behavioral analysis
        neckX = data['neck']["x"][reliable_idx['neck']].reset_index(drop=True)
        neckY = data['neck']["y"][reliable_idx['neck']].reset_index(drop=True)
        neck_full_idx = reliable_idx['neck'].to_numpy()
    
        fps = 30
        dark_room_time = light_room_time = tube_time = latency = 0
        data = []
        data1 = []
        data2 = []
        data3 = []
        data6 = []
        data7 = []
    
        start_i = 0
        cycle = 0
        success_num = 0
        first_success = None
    
        while start_i < len(neckX) and cycle <= 100:
            # Find light room entry
            light_entry = None
            for i in range(start_i, len(neckX)):
                if 0 < neckX.iloc[i] - (neckY.iloc[i] - b1) / k1:
                    light_entry = i
                    data.append(neck_full_idx[i])
                    break
            if light_entry is None:
                break
    
            # Find exit to dark room
            dark_entry = None
            for j in range(light_entry + 1, len(neckX)):
                if abs(neckY.iloc[j] - get_last_valid_y('spacer1_2')) < 50\
                    and neckX.iloc[j] > get_last_valid_x('spacer1_1'):
                        dark_entry = j
                        data1.append(neck_full_idx[j])
                        data2.append(neck_full_idx[j])
                        break
            if dark_entry is None:
                break
            
            # Check if passes through tube
            tube_pass = False
            for k in range(dark_entry + 1, len(neckX)):
                if abs(neckY.iloc[k] - get_last_valid_y('spacer1_2')) < 50\
                    and get_last_valid_x('spacer2_1') < neckX.iloc[k] < get_last_valid_x('spacer1_1'):
                        condition = neckX.iloc[j:k] - get_last_valid_x('tube1')
                        first_idx, last_idx = self.find_first_last(condition > dx3)
                        if first_idx is not None:
                            data6.append(neck_full_idx[j + first_idx - 1])
                        if last_idx is not None:
                            data7.append(neck_full_idx[j + last_idx - 1])
                        data3.append(neck_full_idx[k])
                        tube_pass = True
                        if first_success is None:
                            first_success = neck_full_idx[j]
                        success_num += 1
                        start_i = k + 1
                        break
            if not tube_pass:
                start_i = j + 1
            
            cycle += 1
    
        # === 5. Calculate timings ===
        if len(data) >= 2 and len(data1) >= 2:
            light_room_time += (data1[1] - data[1]) / fps
    
        if len(data1) >= 2:
            for i in range(1, len(data1)):
                try:
                    idx = data2.index(data1[i-1])
                    light_room_time += (data1[i] - data3[idx]) / fps
                except ValueError:
                    continue
    
        if data6 and data7:
            tube_time += sum((d2 - d1) / fps for d1, d2 in zip(data6, data7))
    
        if data2 and data3:
            dark_room_time += sum((d2 - d1) / fps for d1, d2 in zip(data2, data3))
    
        if first_success:
            latency = (first_success - data[0]) / fps
    
        num_dark_room = len(data2)
    
        # === 6. Save results ===
        metrics = {
            'light_room_time': light_room_time,
            'tube_time': tube_time,
            'dark_room_time': dark_room_time,
            'num_dark_room': num_dark_room,
            'latency': latency,
            'success_num': success_num
        }
    
        output_metrics_path = os.path.join(self.save_dir, f"{self.file_prefix.get()}_processed_data.csv")
        pd.DataFrame([metrics]).to_csv(output_metrics_path, index=False)
        print(f"[Result] Saved behavioral metrics to {output_metrics_path}")


    def mark_ttl_on(self):
        self._send_to_arduino('T')
        print('mark on')
    
    def mark_ttl_off(self):
        self._send_to_arduino('t')
    
    def trigger_laser(self):
        self._send_to_arduino('L')
    
    def _send_to_arduino(self, command_char):
        try:
            if self.arduino and self.arduino.is_open:
                self.arduino.write(command_char.encode())
                print(f"Sent to Arduino: {command_char}")
        except Exception as e:
            print(f"Serial communication error: {e}")
        
if __name__ == '__main__':
    root = tk.Tk()
    app = BarrierTrigger(root)
    root.mainloop()