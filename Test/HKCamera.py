# -*- coding: utf-8 -*-
"""
Created on Fri May  9 14:15:19 2025

@author: Pulee
"""

import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt 
from MvCameraControl_class import *

class Camera:
    #初始化
    def __init__(self,camera_index):
        """
        初始化参数
        :param camera_index:相机索引，未装驱动电脑索引从0开始，装了驱动的从1开始
        """
        #设备信息表初始化
        self._deviceList = MV_CC_DEVICE_INFO_LIST()
 
        #设备类型
        self._tlayerType = MV_USB_DEVICE
 
        #相机实例
        self._cam = MvCamera()
 
        #相机参数
        self._stParam = None
 
        #数据包大小
        self._nPayloadSize = None
 
        #数据流
        self._data_buf = None
 
        #相机索引
        self._camera_index = camera_index
 
        #相机型号等打印
        self._Show_info = True
 
        #获取设备信息
        MvCamera.MV_CC_EnumDevices(self._tlayerType, self._deviceList)
        
        #打印设备信息
        if self._Show_info:
            self._print_debug_info()
        
        #打开相机流
        self._open()
    def _print_debug_info(self):
        mvcc_dev_info = cast(self._deviceList.pDeviceInfo[self._camera_index], POINTER(MV_CC_DEVICE_INFO)).contents
        if mvcc_dev_info.nTLayerType == MV_USB_DEVICE:
            print("\n设备列表: [%d]" % self._camera_index)
            strModeName = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chModelName:
                if per == 0:
                    break
                strModeName = strModeName + chr(per)
            print("设备名称: %s" % strModeName)
 
            strSerialNumber = ""
            for per in mvcc_dev_info.SpecialInfo.stUsb3VInfo.chSerialNumber:
                if per == 0:
                    break
                strSerialNumber = strSerialNumber + chr(per)
            print("串行代号: %s" % strSerialNumber)
    def _open(self):
        """
        打开设备
        :return: 
        """
        if int(self._camera_index) >= self._deviceList.nDeviceNum:
            print("索引相机失败!")
            sys.exit()
 
        #创建相机实例
 
        stDeviceList = cast(self._deviceList.pDeviceInfo[int(self._camera_index)], POINTER(MV_CC_DEVICE_INFO)).contents
        ret = self._cam.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print("相机打开错误: 相机索引创建句柄失败! 错误码:[0x%x]" % ret)
            sys.exit()
 
        #打开设备
        ret = self._cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print("相机打开错误: 设备打开失败! 错误码:[0x%x]" % ret)
            sys.exit()
 
        ret = self._cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        if ret != 0:
            print("相机打开错误: 触发模式设置失败! 错误码:[0x%x] ret[0x%x]" % ret)
            sys.exit()
 
        #获取数据包大小
        self._stParam = MVCC_INTVALUE()
        memset(byref(self._stParam), 0, sizeof(MVCC_INTVALUE))
 
        ret = self._cam.MV_CC_GetIntValue("PayloadSize", self._stParam)
        if ret != 0:
            print("相机打开错误: 数据包大小获取失败! 错误码:[0x%x]" % ret)
            sys.exit()
        self._nPayloadSize = self._stParam.nCurValue
 
        # ch:开始取流 | en:Start grab image
        ret = self._cam.MV_CC_StartGrabbing()
        if ret != 0:
            print("取流失败: 开始取流失败! 错误码:[0x%x]" % ret)
            sys.exit()
    def get_img(self):
        """
        获取一帧图像
        :return: 
        """
        #创建图像信息表
        stDeviceList = MV_FRAME_OUT_INFO_EX()
        
        #初始化图像信息表
        memset(byref(stDeviceList), 0, sizeof(stDeviceList))
        
        #创建原始图像信息表
        self._data_buf = (c_ubyte * self._nPayloadSize)()
        
        #采用超时机制获取一帧图片，SDK内部等待直到有数据时返回
        ret = self._cam.MV_CC_GetOneFrameTimeout(byref(self._data_buf), self._nPayloadSize, stDeviceList, 1000)
        if ret == 0:
            # print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (stDeviceList.nWidth, stDeviceList.nHeight, stDeviceList.nFrameNum))
            
            #配置图像参数
            nRGBSize = stDeviceList.nWidth * stDeviceList.nHeight * 3
            stConvertParam = MV_SAVE_IMAGE_PARAM_EX()
            stConvertParam.nWidth = stDeviceList.nWidth
            stConvertParam.nHeight = stDeviceList.nHeight
            stConvertParam.pData = self._data_buf
            stConvertParam.nDataLen = stDeviceList.nFrameLen
            stConvertParam.enPixelType = stDeviceList.enPixelType
            stConvertParam.nImageLen = stConvertParam.nDataLen
            stConvertParam.nJpgQuality = 70
            stConvertParam.enImageType = MV_Image_Jpeg
            stConvertParam.pImageBuffer = (c_ubyte * nRGBSize)()
            stConvertParam.nBufferSize = nRGBSize
            # ret = cam.MV_CC_ConvertPixelType(stConvertParam)
            # print(stConvertParam.nImageLen)
            
            #覆盖上一帧图像
            ret = self._cam.MV_CC_SaveImageEx2(stConvertParam)
            if ret != 0:
                print("convert pixel fail ! ret[0x%x]" % ret)
                del self._data_buf
                sys.exit()
            
            #获取图像信息
            img_buff = (c_ubyte * stConvertParam.nImageLen)()
            cdll.msvcrt.memcpy(byref(img_buff), stConvertParam.pImageBuffer, stConvertParam.nImageLen)
 
            # 将 ctypes 数组转换为 NumPy 数组
            _img_array = np.frombuffer(img_buff, dtype=np.uint8)
 
            # 使用 cv2.imdecode 解码图像
            _image = cv2.imdecode(_img_array, cv2.IMREAD_COLOR)
 
            return _image
        
    def _close(self):
 
        ret = self._cam.MV_CC_StopGrabbing()
        if ret != 0:
            print("相机关闭失败: 停止取流失败! 错误码:[0x%x]" % ret)
            del self._data_buf
            sys.exit()
 
 
        ret = self._cam.MV_CC_CloseDevice()
        if ret != 0:
            print("相机关闭失败: 设别关闭失败! 错误码:[0x%x]" % ret)
            del self._data_buf
            sys.exit()
 
 
        ret = self._cam.MV_CC_DestroyHandle()
        if ret != 0:
            print("相机关闭失败: 句柄销毁失败! 错误码:[0x%x]" % ret)
            del self._data_buf
            sys.exit()
 
        del self._data_buf
        ...
        
def main():
    # 初始化matplotlib交互模式
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots()  # 创建绘图对象
    image_display = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))  # 初始化空白图像
    plt.show(block=False)  # 非阻塞显示
    
    camera = Camera(0)
    try:
        while True:
            img = camera.get_img()
            
            if img is None:
                continue
                
            # 将BGR转换为RGB（Matplotlib需要RGB格式）
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 更新图像数据
            image_display.set_data(img_rgb)
            ax.draw_artist(image_display)
            fig.canvas.blit(ax.bbox)
            fig.canvas.flush_events()  # 刷新显示
            
            # 通过matplotlib的关闭按钮退出
            if not plt.fignum_exists(fig.number):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        plt.close()  # 关闭图像窗口
        camera._close()  # 确保释放相机资源

if __name__ == '__main__':
    main()