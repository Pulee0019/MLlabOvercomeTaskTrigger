# -*- coding: utf-8 -*-
"""
Created on Fri May 16 16:06:59 2025

@author: Pulee
"""

import os
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
from ctypes import *
import numpy as np
import pandas as pd
import cv2
import queue
import serial
from pathlib import Path
from functools import partial
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.core import predict
from deeplabcut.utils import auxiliaryfunctions
from MvCameraControl_class import *
from CameraParams_header import *
from MvErrorDefine_const import *

PixelType_Mono8 = 17301505

class HKCamera:
    def __init__(self, root, arduino=None):
        self.root = root
        self.arduino = arduino
        
        self.cam = MvCamera()
        self.device_list = MV_CC_DEVICE_INFO_LIST()
        self.tlayer_type = MV_USB_DEVICE
        self.recording = False
        self.previewing = False
        self.preview_callback = None
        self.last_frame_data = None
        self.save_path = None
        self.analysis_queue = queue.Queue(maxsize=5)
        self.running = False

        self.width = 1280
        self.height = 1024
        
        self.frame_rate = 30  # Target video record frame rate
        self.sample_rate = 201.7  # Hardware image capture frame rate
        self.pixel_type = PixelType_Mono8

        # DLC loading
        self.dlc_loaded = False
        self.pose_callback = None
        self.bodyparts = []
        self.skeleton = []
        self.dlc_cfg = None
        self.dlc_session = None
        self.inputs = None
        self.outputs = None

        self.trigger_flag = 0
        self.trigger_counter = 0
        self.triggerFrame = 0
        self.trigger_log = []
        self.barrier_idx = None
        self.mouth_idx = None
        self.start_time = time.time()

        # Buffer for video frames
        self.frame_buffer = []
        self.max_buffer_size = 5
        
    def load_dlc_model(self, config_path):
        cfg = auxiliaryfunctions.read_config(config_path)
        shuffle = 1
        trainingsetindex = 0
        trainFraction = cfg['TrainingFraction'][trainingsetindex]
        modelfolder = os.path.join(cfg['project_path'], str(auxiliaryfunctions.get_model_folder(trainFraction, shuffle, cfg)))
        path_test_config = os.path.join(modelfolder, 'test', 'pose_cfg.yaml')
        self.dlc_cfg = load_config(path_test_config)

        snapshots = auxiliaryfunctions.get_snapshots_from_folder(Path(modelfolder) / "train")
        snapshot_name = snapshots[-1]
        self.dlc_cfg["init_weights"] = os.path.join(modelfolder, "train", snapshot_name)
        self.dlc_cfg["batch_size"] = 1
        
        trainingsiterations = (self.dlc_cfg["init_weights"].split(os.sep)[-1]).split("-")[-1]
        DLCscorer = auxiliaryfunctions.GetScorerName(cfg, 
                                                     shuffle, 
                                                     trainFraction, 
                                                     trainingsiterations=trainingsiterations
                                                     )
        if isinstance(DLCscorer, tuple):
            DLCscorer = DLCscorer[0]

        self.bodyparts = cfg['bodyparts']
        self.skeleton = cfg['skeleton']
        
        self.barrier_idx = self.dlc_cfg['all_joints_names'].index('barrier')
        self.mouth_idx = self.dlc_cfg['all_joints_names'].index('mouth')

        self.dlc_session, self.inputs, self.outputs = predict.setup_pose_prediction(self.dlc_cfg)
        self.dlc_loaded = True
        
        self.pdindex = pd.MultiIndex.from_product([[DLCscorer], 
                                              self.dlc_cfg['all_joints_names'], 
                                              ['x', 'y', 'likelihood']], 
                                             names=['scorer', 'bodyparts', 'coords']
                                             )
        
        print(f"[INFO] Loaded DLC model: {snapshot_name}")

    def open(self):
        ret = self.cam.MV_CC_EnumDevices(self.tlayer_type, self.device_list)
        if ret != MV_OK or self.device_list.nDeviceNum == 0:
            raise RuntimeError("No camera found.")
    
        dev_info = self.device_list.pDeviceInfo[0].contents
        self.cam.MV_CC_CreateHandle(dev_info)
        self.cam.MV_CC_OpenDevice()
    
        self.cam.MV_CC_SetEnumValue("PixelFormat", self.pixel_type)
    
        width_info = MVCC_INTVALUE()
        height_info = MVCC_INTVALUE()
        self.cam.MV_CC_GetIntValue("WidthMax", width_info)
        self.cam.MV_CC_GetIntValue("HeightMax", height_info)
    
        offsetx_info = MVCC_INTVALUE()
        offsety_info = MVCC_INTVALUE()
        self.cam.MV_CC_GetIntValue("OffsetX", offsetx_info)
        self.cam.MV_CC_GetIntValue("OffsetY", offsety_info)
    
        self.cam.MV_CC_SetIntValue("OffsetX", offsetx_info.nMin)
        self.cam.MV_CC_SetIntValue("OffsetY", offsety_info.nMin)
    
        self.cam.MV_CC_SetIntValue("Width", width_info.nCurValue)
        self.cam.MV_CC_SetIntValue("Height", height_info.nCurValue)
    
        self.width = width_info.nCurValue
        self.height = height_info.nCurValue
    
        self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", self.sample_rate)
    
        self.cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    
        self.cam.MV_CC_SetGrabStrategy(MV_GrabStrategy_LatestImages)
    
        self.cam.MV_CC_StartGrabbing()
    
        print(f"[INFO] Opened camera with resolution {self.width} x {self.height}, offset 0,0")


    def close(self):
        try:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
            self.cam.MV_CC_DestroyHandle()
            print("[INFO] Camera successfully closed.")
        except Exception as e:
            print(f"[WARN] Error closing camera: {e}")

    
    def draw_pose(self, img, keypoints, bodyparts, skeleton):
        for i, (x, y, p) in enumerate(keypoints):
            if p > 0.5:
                cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
                cv2.putText(img, bodyparts[i], (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        
        for joint1, joint2 in skeleton:
            i1 = bodyparts.index(joint1)
            i2 = bodyparts.index(joint2)
            if keypoints[i1][2] > 0.5 and keypoints[i2][2] > 0.5:
                pt1 = (int(keypoints[i1][0]), int(keypoints[i1][1]))
                pt2 = (int(keypoints[i2][0]), int(keypoints[i2][1]))
                cv2.line(img, pt1, pt2, (255, 0, 0), 2)
        return img

    def start_preview(self, callback):
        self.previewing = True
        self.preview_callback = callback
        threading.Thread(target=self._preview_loop, daemon=True).start()

    def stop_preview(self):
        self.previewing = False

    def _preview_loop(self):
        stFrame = MV_FRAME_OUT()
        while self.previewing:
            ret = self.cam.MV_CC_GetImageBuffer(stFrame, 1000)
            if ret == MV_OK:
                buf = string_at(stFrame.pBufAddr, stFrame.stFrameInfo.nFrameLen)
                self.last_frame_data = buf
                img_np = np.frombuffer(buf, dtype=np.uint8).reshape((stFrame.stFrameInfo.nHeight, stFrame.stFrameInfo.nWidth))

                pil_img = Image.fromarray(img_np)
                self.cam.MV_CC_FreeImageBuffer(stFrame)
                
                if self.preview_callback:
                    if self.preview_callback:
                        self.root.after(0, lambda: self.preview_callback(pil_img))
            else:
                time.sleep(0.01)

    def start_record(self, file_path):
        param = MV_CC_RECORD_PARAM()
        param.enPixelType = self.pixel_type
        param.nWidth = self.width
        param.nHeight = self.height
        param.fFrameRate = self.frame_rate  # Recording frame rate
        param.nBitRate = 8192
        param.enRecordFmtType = 1
        param.strFilePath = c_char_p(file_path.encode('utf-8'))
        self.cam.MV_CC_StartRecord(param)
        self.recording = True
        self.save_path = file_path
        self.running = True
        threading.Thread(target=self._acquisition_loop, daemon=True).start()
        threading.Thread(target=self._analysis_loop, daemon=True).start()
        threading.Thread(target=self._record_loop, daemon=True).start()
    
    def _acquisition_loop(self):
        stFrame = MV_FRAME_OUT()
        while self.running:
            ret = self.cam.MV_CC_GetImageBuffer(stFrame, 1000)
            if ret == MV_OK:
                buf = string_at(stFrame.pBufAddr, stFrame.stFrameInfo.nFrameLen)
                img_np = np.frombuffer(buf, dtype=np.uint8).reshape((self.height, self.width))
                self.last_frame_data = buf
                self.cam.MV_CC_FreeImageBuffer(stFrame)
                
                if not self.analysis_queue.full():
                    if not self.dlc_loaded:
                        print("[WARN] DLC not loaded. Skipping frame.")
                        continue
                    self.analysis_queue.put(img_np.copy())
    
    def _analysis_loop(self):
        counter = 0
        flag = 0
        FirstStartTime = time.time()
        PredictedData = []
        barrier_idx = self.barrier_idx
        mouth_idx = self.mouth_idx
        startTime = time.time()
        triggerFrame = 0
        triggerTimesA = []
        triggerTimesT = []
        allDuration = []
        while self.running:
            try:
                frame = self.analysis_queue.get(timeout=1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                pose = predict.getpose(rgb, self.dlc_cfg, self.dlc_session, self.inputs, self.outputs)
                PredictedData.append(pose.flatten())
                counter += 1
                
                # Live display update
                frame_copy = frame.copy()  # Copy the original frame
                vis_img = self.draw_pose(frame_copy, pose, self.bodyparts, self.skeleton)
                pil_img = Image.fromarray(vis_img)
                
                if hasattr(self, 'pose_callback') and self.pose_callback:
                    self.root.after(0, lambda: self.pose_callback(pil_img))
                
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
            except queue.Empty:
                continue
        df_triggers = pd.DataFrame({
            "ActualTime": triggerTimesA,
            "TheoreticalTime": triggerTimesT,
            "Duration": allDuration
        })
        trigger_path = os.path.join(self.save_path.replace('.avi', '_triggers.csv'))
        df_triggers.to_csv(trigger_path, index=False)
        print(f"[INFO] Trigger log saved to {trigger_path}")
        
        df_pred = pd.DataFrame(PredictedData, columns=self.pdindex)
        pred_path = self.save_path.replace('.avi', '_predicted.csv')
        df_pred.to_csv(pred_path, index=False)
        print(f"[INFO] Pose data saved to {pred_path}")


    def _record_loop(self):
        stInput = MV_CC_INPUT_FRAME_INFO()
        self.frame_counter = 0
        start_time = time.time()
        self.mark_ttl_on()
        target_interval_t = 1.0 / self.frame_rate  # Target interval for recording frame rate

        while self.recording:
            loop_start = time.time()
    
            if self.last_frame_data is None:
                time.sleep(0.01)
                continue
    
            stInput.pData = cast(c_char_p(self.last_frame_data), POINTER(c_ubyte))
            stInput.nDataLen = len(self.last_frame_data)
            self.cam.MV_CC_InputOneFrame(stInput)
            self.frame_counter += 1
    
            if self.frame_counter % 100 == 0:
                elapsed = time.time() - start_time
                fps = self.frame_counter / elapsed
                self.gui_fps_var.set(f"FPS: {fps:.2f}")
    
            target_interval = self.frame_counter * target_interval_t - (time.time() - start_time)
            if target_interval > 0:
                time.sleep(target_interval)
                
    def stop_record(self):
        self.running = False
        self.recording = False
        self.mark_ttl_off()
        self.cam.MV_CC_StopRecord()
    
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

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HK Camera GUI")

        self.arduino_port = "COM10"
        try:
            self.arduino = serial.Serial(self.arduino_port, 9600, timeout=1)
            time.sleep(2)
        except serial.SerialException as e:
            messagebox.showerror("Arduino Error", f"Failed to open {self.arduino_port}: {e}")
            self.arduino = None


        self.camera = HKCamera(root, arduino=self.arduino)
        self.camera.pose_callback = self.update_pose_image 

        self.save_path = tk.StringVar()
        self.dlc_path = tk.StringVar()

        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0)

        ttk.Button(frm, text="Browse", command=self.browse_path).grid(row=0, column=0, sticky="W")
        ttk.Entry(frm, textvariable=self.save_path, width=20).grid(row=0, column=1)
        ttk.Button(frm, text="Load DLC", command=self.load_dlc).grid(row=1, column=0, sticky="W")
        ttk.Entry(frm, textvariable=self.dlc_path, width=20).grid(row=1, column=1)
        ttk.Button(frm, text="Open Camera", command=self.open_camera).grid(row=2, column=0)
        ttk.Button(frm, text="Exit", command=self.exit).grid(row=2, column=1)
        ttk.Button(frm, text="Start Preview", command=self.start_preview).grid(row=3, column=0)
        ttk.Button(frm, text="Stop Preview", command=self.stop_preview).grid(row=3, column=1)
        ttk.Button(frm, text="Start Record", command=self.start_record).grid(row=4, column=0)
        ttk.Button(frm, text="Stop Record", command=self.stop_record).grid(row=4, column=1)

        self.recording_time_label = tk.Label(self.root, text="", font=("Helvetica", 16), fg="red")
        self.recording_time_label.grid(row=6, column=0, sticky="se", padx=10, pady=10)

        self.raw_label = tk.Label(root)
        self.raw_label.grid(row=5, column=0, padx=10, pady=10)
        
        self.pose_label = tk.Label(root)
        self.pose_label.grid(row=5, column=1, padx=10, pady=10)

        self.fps_label = ttk.Label(self.root, text="FPS: --", font=("Helvetica", 12))
        self.fps_label.grid(row=7, column=0, sticky="w", padx=10)
        self.camera.gui_fps_var = tk.StringVar(value="FPS: --")
        self.fps_label.config(textvariable=self.camera.gui_fps_var)
        self.label_status = ttk.Label(self.root, text="Ready")
        self.label_status.grid(row=8, column=0, columnspan=2)
    
    def on_closing(self):
        try:
            self.camera.stop_record()
            self.camera.stop_preview()
            self.camera.close()
            if self.arduino and self.arduino.is_open:
                self.arduino.close()
                print("Arduino port closed.")
        except Exception as e:
            print(f"[ERROR] Exception during closing: {e}")
        finally:
            self.root.destroy()
            
    def load_dlc(self):
        config_path = filedialog.askopenfilename(filetypes=[("YAML Files", "*.yaml")])
        if config_path:
            self.dlc_path.set(config_path)
            self.camera.load_dlc_model(config_path)

    def browse_path(self):
        path = filedialog.asksaveasfilename(defaultextension=".avi")
        if path:
            self.save_path.set(path)

    def open_camera(self):
        try:
            self.camera.open()
            messagebox.showinfo("Success", "Camera opened.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def start_preview(self):
        self.camera.start_preview(self.update_image)
        
    def stop_preview(self):
        if self.camera.previewing:
            self.camera.stop_preview()
            print("[INFO] Preview stopped.")
        else:
            print("[INFO] No preview is running.")

    def update_image(self, pil_image):
        target_size = (640, 480)
        padded = ImageOps.pad(pil_image.convert("L"), target_size, color=0, centering=(0.5, 0.5))
        imgtk = ImageTk.PhotoImage(padded)
        self.raw_label.after(0, lambda: self._update_label(self.raw_label, imgtk))
    
    def update_pose_image(self, pil_image):
        target_size = (640, 480)
        padded = ImageOps.pad(pil_image.convert("L"), target_size, color=0, centering=(0.5, 0.5))
        imgtk = ImageTk.PhotoImage(padded)
        self.pose_label.after(0, lambda: self._update_label(self.pose_label, imgtk))


    def _update_label(self, label, imgtk):
        label.imgtk = imgtk
        label.configure(image=imgtk)

    def start_record(self):
        if not self.save_path.get():
            messagebox.showerror("Error", "Please set save path.")
            return
        self.camera.start_record(self.save_path.get())
        self.recording_start_time = time.time()
        self.frame_counter = 0
        self.update_recording_time()
    
    def update_recording_time(self):
        if self.camera.recording:
            elapsed = time.time() - self.recording_start_time
            mins, secs = divmod(int(elapsed), 60)
            time_str = f"{mins:02}:{secs:02}"
            self.recording_time_label.config(text=f"Recording: {time_str}")
            self.root.after(1000, self.update_recording_time)
        else:
            self.recording_time_label.config(text="")

    def stop_record(self):
        self.camera.stop_record()
        self.recording_time_label.config(text="")
        self.label_status.config(text="Recording stopped")

    def exit(self):
        self.camera.stop_preview()
        self.camera.close()
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("Arduino port closed.")
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop() 