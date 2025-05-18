# Updated Barrier Trigger Application - Getters Fixed (no double byref)

import os
import cv2
import time
import serial
import tkinter as tk
import numpy as np
import pandas as pd
import threading
from ctypes import *
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageTk
from tkinter import ttk, filedialog, messagebox
from MvCameraControl_class import *
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.core import predict
from deeplabcut.utils import auxiliaryfunctions

# Pixel type constants
PixelType_Gvsp_Mono8 = 0x01080001
PixelType_Gvsp_RGB8_Packed = 0x02180014
PixelType_Gvsp_BGR8_Packed = 0x02180015

class MVCC_INTVALUE(Structure):
    _fields_ = [
        ("nCurValue", c_uint),
        ("nMin", c_uint),
        ("nMax", c_uint),
        ("nInc", c_uint)
    ]

class MVCC_FLOATVALUE(Structure):
    _fields_ = [
        ("fCurValue", c_float),
        ("fMin", c_float),
        ("fMax", c_float),
        ("fInc", c_float)
    ]

class HikCameraWrapper:
    def __init__(self):
        self.cam = MvCamera()
        self.handle_created = False

    def get_supported_resolution_range(self):
        """
        Query the camera for the maximum supported width and height.
        Return a list of common resolutions filtered by those limits.
        """
        width_info = MVCC_INTVALUE()
        height_info = MVCC_INTVALUE()
        
        # Get max supported width and height from camera
        self.cam.MV_CC_GetIntValue("WidthMax", width_info)
        self.cam.MV_CC_GetIntValue("HeightMax", height_info)

        width_max = width_info.nCurValue
        height_max = height_info.nCurValue

        # Define standard/common resolution presets
        common_res = [
            (3840, 2160), (2560, 1440), (1920, 1080), (1600, 1200),
            (1280, 1024), (1280, 720), (1024, 768), (800, 600),
            (640, 480), (320, 240)
        ]

        # Keep only resolutions that fit within the camera's range
        valid_res = [(w, h) for w, h in common_res if w <= width_max and h <= height_max]
        return valid_res

    def open_camera(self, index):
        device_list = MV_CC_DEVICE_INFO_LIST()
        MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
        if index >= device_list.nDeviceNum:
            raise RuntimeError("Invalid camera index")
        dev_info = cast(device_list.pDeviceInfo[index], POINTER(MV_CC_DEVICE_INFO)).contents
        if self.handle_created:
            self.close()
        ret = self.cam.MV_CC_CreateHandle(dev_info)
        if ret != 0:
            raise RuntimeError(f"Create handle failed: 0x{ret:x}")
        self.handle_created = True
        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise RuntimeError(f"Open device failed: 0x{ret:x}")
        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        self.cam.MV_CC_StartGrabbing()

    def set_resolution(self, width, height):
        self.cam.MV_CC_SetIntValue("Width", width)
        self.cam.MV_CC_SetIntValue("Height", height)

    def get_current_resolution(self):
        width_val = MVCC_INTVALUE()
        height_val = MVCC_INTVALUE()
        ret1 = self.cam.MV_CC_GetIntValue("Width", width_val)
        ret2 = self.cam.MV_CC_GetIntValue("Height", height_val)
        if ret1 != 0 or ret2 != 0:
            raise RuntimeError("Failed to get width or height from camera.")
        return width_val.nCurValue, height_val.nCurValue

    def get_exposure(self):
        expo = MVCC_FLOATVALUE()
        ret = self.cam.MV_CC_GetFloatValue("ExposureTime", expo)
        if ret != 0:
            raise RuntimeError("Failed to get exposure time.")
        return expo.fCurValue

    def get_fps(self):
        fps_val = MVCC_FLOATVALUE()
        ret = self.cam.MV_CC_GetFloatValue("ResultingFrameRate", fps_val)
        if ret != 0:
            raise RuntimeError("Failed to get FPS from camera.")
        return fps_val.fCurValue

    def set_exposure(self, exposure_time_us):
        ret = self.cam.MV_CC_SetFloatValue("ExposureTime", float(exposure_time_us))
        if ret != 0:
            raise RuntimeError("Failed to set exposure time.")

    def set_fps(self, fps_val):
        ret = self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(fps_val))
        if ret != 0:
            raise RuntimeError("Failed to set frame rate.")

    def get_frame(self):
        frame = MV_FRAME_OUT()
        memset(byref(frame), 0, sizeof(frame))
        ret = self.cam.MV_CC_GetImageBuffer(frame, 1000)
        if ret != 0:
            return None
        size = frame.stFrameInfo.nFrameLen
        w = frame.stFrameInfo.nWidth
        h = frame.stFrameInfo.nHeight
        pixel_type = frame.stFrameInfo.enPixelType
        buffer = cast(frame.pBufAddr, POINTER(c_ubyte * size)).contents
        img_array = np.frombuffer(buffer, dtype=np.uint8)
        if pixel_type == PixelType_Gvsp_Mono8:
            img = img_array.reshape((h, w))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif pixel_type in [PixelType_Gvsp_RGB8_Packed, PixelType_Gvsp_BGR8_Packed]:
            img = img_array.reshape((h, w, 3))
        else:
            print(f"[Warning] Unsupported pixel format: {pixel_type}")
            return None
        self.cam.MV_CC_FreeImageBuffer(frame)
        return img

    def close(self):
        if self.handle_created:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            self.handle_created = False

# === Main BarrierTrigger GUI ===
class BarrierTrigger:
    def __init__(self, root):
        self.root = root
        self.root.title("Barrier Trigger")
        self.camera = HikCameraWrapper()
        self.previewing = False
        self.recording = False

        self.save_dir = os.getcwd()
        self.file_prefix = tk.StringVar(value="test")
        self.selected_resolution = tk.StringVar()
        self.video_writer = None
        self.recording_start_time = None

        self.setup_ui()
        self.arduino = serial.Serial("COM10", 9600, timeout=1)
        time.sleep(2)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.config_path = r"D:/Expriment/DLC/ycyy/tube-ycyy-hes-2025-04-28/config.yaml"
        cfg = auxiliaryfunctions.read_config(self.config_path)
        trainFraction = cfg["TrainingFraction"][0]
        modelfolder = os.path.join(cfg["project_path"], auxiliaryfunctions.get_model_folder(trainFraction, 1, cfg))
        snapshot_name = auxiliaryfunctions.get_snapshots_from_folder(Path(modelfolder)/"train")[-1]
        dlc_cfg = load_config(str(Path(modelfolder)/"test"/"pose_cfg.yaml"))
        dlc_cfg["init_weights"] = os.path.join(modelfolder, "train", snapshot_name)
        dlc_cfg["batch_size"] = 1
        self.dlc_cfg = dlc_cfg
        self.cfg = cfg
        self.bodyparts = cfg['bodyparts']
        self.skeleton = cfg.get('skeleton', [])
        self.sess, self.inputs, self.outputs = predict.setup_pose_prediction(dlc_cfg)

    def setup_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0)
    
        # Save directory selection button
        ttk.Button(frm, text="Set Save Directory", command=self.select_directory)\
            .grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
    
        # Filename prefix input
        ttk.Label(frm, text="Prefix:").grid(row=1, column=0)
        ttk.Entry(frm, textvariable=self.file_prefix).grid(row=1, column=1)
    
        # Open camera temporarily to fetch supported resolutions
        self.camera.open_camera(0)
        supported = self.camera.get_supported_resolution_range()
        self.camera.close()
    
        # Fill resolution dropdown and set default
        res_list = [f"{w}x{h}" for w, h in supported]
        if res_list:
            self.selected_resolution.set(res_list[0])  # Set default value
    
        # Resolution selection dropdown
        ttk.Label(frm, text="Resolution:").grid(row=2, column=0)
        res_combo = ttk.Combobox(frm, textvariable=self.selected_resolution, values=res_list, state="readonly")
        res_combo.grid(row=2, column=1)
        if res_list:
            res_combo.current(0)
    
        # Preview and Record buttons
        ttk.Button(frm, text="Start Preview", command=self.start_preview).grid(row=3, column=0)
        ttk.Button(frm, text="Stop Preview", command=self.stop_preview).grid(row=3, column=1)
        ttk.Button(frm, text="Start Record", command=self.start_recording).grid(row=4, column=0)
        ttk.Button(frm, text="Stop Record", command=self.stop_recording).grid(row=4, column=1)
    
        # Exposure and FPS inputs
        ttk.Label(frm, text="Exposure (us):").grid(row=5, column=0)
        self.exposure_var = tk.StringVar(value="1000")
        ttk.Entry(frm, textvariable=self.exposure_var).grid(row=5, column=1)
    
        ttk.Label(frm, text="FPS:").grid(row=6, column=0)
        self.fps_var = tk.StringVar(value="30")
        ttk.Entry(frm, textvariable=self.fps_var).grid(row=6, column=1)
    
        # Status label
        self.label_status = ttk.Label(frm, text="Ready")
        self.label_status.grid(row=7, column=0, columnspan=2)
    
        # Image preview canvas and timer
        self.canvas = tk.Label(self.root)
        self.canvas.grid(row=0, column=1)
        self.recording_timer = tk.Label(self.root, font=("Helvetica", 20), fg="red")
        self.recording_timer.grid(row=1, column=1, sticky="e", padx=20)
    
    def select_directory(self):
        path = filedialog.askdirectory()
        if path:
            self.save_dir = path
            
    def create_save_dir(self):
        """
        Create a subfolder under the selected save path using today's date.
        """
        date_folder = time.strftime("%Y-%m-%d")
        save_path = os.path.join(self.save_dir, date_folder)
        os.makedirs(save_path, exist_ok=True)
        return save_path
            
    def pad_to_640x480(self, frame):
        """Pad the input frame to 640x480 with black borders."""
        target_w, target_h = 640, 480
        h, w = frame.shape[:2]
        top = (target_h - h) // 2 if target_h > h else 0
        bottom = target_h - h - top if target_h > h else 0
        left = (target_w - w) // 2 if target_w > w else 0
        right = target_w - w - left if target_w > w else 0
        padded_frame = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return cv2.resize(padded_frame, (640, 480))

    def start_preview(self):
        # Get and validate selected resolution string
        res_str = self.selected_resolution.get()
        if 'x' not in res_str:
            messagebox.showerror("Error", "Please select a valid resolution before preview.")
            return
    
        # Parse width and height from resolution string
        try:
            w, h = map(int, res_str.split('x'))
        except ValueError:
            messagebox.showerror("Error", f"Invalid resolution format: {res_str}")
            return
    
        # Open and configure camera
        self.camera.open_camera(0)
        self.camera.set_resolution(w, h)
        self.camera.set_exposure(float(self.exposure_var.get()))
        self.camera.set_fps(float(self.fps_var.get()))
        self.previewing = True
        self.preview_loop()
    
        # Print camera info
        current_expo = self.camera.get_exposure()
        current_fps = self.camera.get_fps()
        print(f"[INFO] Current Exposure: {current_expo} us, FPS: {current_fps}")

    def preview_loop(self):
        if not self.previewing:
            return
        frame = self.camera.get_frame()
        if frame is not None:
            padded = self.pad_to_640x480(frame)
            img = ImageTk.PhotoImage(Image.fromarray(padded), master=self.root)
            self.canvas.imgtk = img
            self.canvas.config(image=img)
        self.root.after(20, self.preview_loop)

    def stop_preview(self):
        self.previewing = False
        self.camera.close()

    def start_recording(self):
        # Ensure camera is opened if not already
        if not self.camera.handle_created:
            self.camera.open_camera(0)
            w, h = map(int, self.selected_resolution.get().split('x'))
            self.camera.set_resolution(w, h)
            self.camera.set_exposure(float(self.exposure_var.get()))
            self.camera.set_fps(float(self.fps_var.get()))
    
        self.recording = True
        self.label_status.config(text="Recording...")
        self.recording_start_time = time.time()
        self.update_recording_timer()
    
        save_dir = self.create_save_dir()
        filename = f"{self.file_prefix.get()}_{time.strftime('%H%M%S')}.avi"
        full_path = os.path.join(save_dir, filename)
    
        try:
            w, h = self.camera.get_current_resolution()
        except RuntimeError as e:
            messagebox.showerror("Camera Error", f"Failed to get resolution:\n{str(e)}")
            return
    
        fps = self.camera.get_fps()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(full_path, fourcc, fps, (w, h))
    
        threading.Thread(target=self.record_loop, args=(save_dir,)).start()
    
    def record_loop(self, save_dir):
       results = []
       counter = 0
       flag = 0
       triggerFrame = 0
       triggerTimesA = []
       triggerTimesT = []
       allDuration = []
       FirstStartTime = time.time()
       pbar = tqdm()
       self.mark_ttl_on()
       self.frame_counter = 0
       FirstStartTime = time.time()
       while self.recording:
           startTime = time.time()
           frame = self.camera.get_frame()
           if frame is None:
               continue
           self.video_writer.write(frame)
           
           # Compute Actual FPS
           self.frame_counter += 1
           if self.frame_counter % 100 == 0:
               elapsed = time.time() - self.recording_start_time
               measured_fps = self.frame_counter / elapsed
               print(f"[Measure] Actual FPS during recording: {measured_fps:.2f}")
               
           frame_rgb = frame[..., ::-1]
           pose = predict.getpose(frame_rgb, self.dlc_cfg, self.sess, self.inputs, self.outputs)
           results.append(pose.flatten())
           self.update_canvas_with_pose(frame.copy(), pose)
           counter += 1
           pbar.update(1)
           
           barrier_idx = self.dlc_cfg['all_joints_names'].index('barrier')
           mouth_idx = self.dlc_cfg['all_joints_names'].index('mouth')

           if flag == 0 and counter >= 2:
               barrier = results[counter-1][3*barrier_idx:3*(barrier_idx+1)]
               mouth = results[counter-1][3*mouth_idx:3*(mouth_idx+1)]
               prev_mouth = results[counter-2][3*mouth_idx]
               if 0 < barrier[0] - mouth[0] < 15 and abs(mouth[1] - barrier[1]) < 15 and mouth[2] > 0.99 and barrier[2] > 0.99 and prev_mouth < mouth[0]:
                   self.arduino.write(b'L')
                   print("[Trigger] Laser ON")
                   endTime = time.time()
                   triggerTimesA.append(endTime - FirstStartTime)
                   triggerTimesT.append(counter / 30)
                   allDuration.append(endTime - startTime)
                   triggerFrame = counter
                   flag = 1
                   print(counter)

           if flag == 1 and counter - triggerFrame >= 10:
               valid_following = 0
               for j in range(triggerFrame + 1, counter):
                   mouth_follow = results[j][3*mouth_idx:3*(mouth_idx+1)]
                   if mouth_follow[2] > 0.99 and results[triggerFrame][3*mouth_idx] - mouth_follow[0] > 10:
                       valid_following += 1
               if valid_following >= 10:
                   flag = 0
       pbar.close()
       df = pd.DataFrame(np.array(results), columns=pd.MultiIndex.from_product(
           [[auxiliaryfunctions.GetScorerName(self.cfg, 1, self.cfg["TrainingFraction"][0])],
            self.dlc_cfg['all_joints_names'], ['x', 'y', 'likelihood']]))
       df.to_csv(os.path.join(save_dir, f"{self.file_prefix.get()}_pose.csv"), index=False)
       pd.DataFrame({"TriggerTime": triggerTimesA, "TriggerFrame": triggerTimesT, "Latency": allDuration}).to_csv(
           os.path.join(save_dir, f"{self.file_prefix.get()}_triggerlog.csv"), index=False)
       
    def update_recording_timer(self):
        if self.recording and self.recording_start_time:
            elapsed = int(time.time() - self.recording_start_time)
            mins, secs = divmod(elapsed, 60)
            self.recording_timer.config(text=f"{mins:02}:{secs:02}")
            self.root.after(1000, self.update_recording_timer)
        else:
            self.recording_timer.config(text="")
    
    def update_canvas_with_pose(self, frame, pose):
        joints = {}
        for idx, name in enumerate(self.bodyparts):
            x, y, l = pose[idx]
            if l > 0.7:
                joints[name] = (int(x), int(y))
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
    
        for pair in self.skeleton:
            if pair[0] in joints and pair[1] in joints:
                cv2.line(frame, joints[pair[0]], joints[pair[1]], (255, 0, 0), 2)
    
        padded_frame = self.pad_to_640x480(frame)
        img = ImageTk.PhotoImage(Image.fromarray(padded_frame), master=self.root)
        self.canvas.imgtk = img
        self.canvas.config(image=img)

    def stop_recording(self):
        self.recording = False
        if self.video_writer:
            self.video_writer.release()
        self.label_status.config(text="Stopped")

    def mark_ttl_on(self):
        self._send_to_arduino('T')
        # print('mark on')

    def mark_ttl_off(self):
        self._send_to_arduino('t')

    def trigger_laser(self):
        self._send_to_arduino('L')

    def _send_to_arduino(self, command_char):
        try:
            if self.arduino and self.arduino.is_open:
                self.arduino.write(command_char.encode())
                # print(f"Sent to Arduino: {command_char}")
        except Exception as e:
            print(f"Serial communication error: {e}")

    def on_closing(self):
        self.stop_preview()
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = BarrierTrigger(root)
    root.mainloop()