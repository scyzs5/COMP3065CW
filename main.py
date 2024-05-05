# import cv2
# import numpy as np
# import tkinter as tk
# from tkinter import filedialog
#
# def detect_faces(video_path, output_index):
#     # 加载人脸检测模型
#     modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
#     configFile = "deploy.prototxt"
#     net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
#
#     # 打开视频文件
#     cap = cv2.VideoCapture(video_path)
#
#     # 获取视频的帧率、宽度和高度
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#     # 创建 VideoWriter 对象
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     output_filename = 'output{}.avi'.format(output_index)
#     out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # 将帧转换为 blob 格式
#         blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
#
#         # 输入 blob 到网络中进行人脸检测
#         net.setInput(blob)
#         detections = net.forward()
#
#         # 遍历检测结果
#         for i in range(detections.shape[2]):
#             confidence = detections[0, 0, i, 2]
#
#             # 当置信度大于阈值时进行绘制
#             if confidence > 0.5:
#                 # 获取检测框的坐标
#                 box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
#                 (x, y, w, h) = box.astype(int)
#
#                 # 绘制检测框和置信度
#                 cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
#                 text = "Confidence: {:.2f}".format(confidence)
#                 cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#         # 将处理过的帧写入输出视频文件
#         out.write(frame)
#
#     # 清理
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#
#     print("视频处理完成并保存为{}".format(output_filename))
#
# def select_video():
#     # 打开文件选择对话框
#     filepath = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
#     video_entry.delete(0, tk.END)
#     video_entry.insert(0, filepath)
#
# def start_detection():
#     video_path = video_entry.get()
#     global output_index
#     detect_faces(video_path, output_index)
#     output_index += 1
#
# def exit_app():
#     root.destroy()
#
# # 创建Tkinter窗口
# root = tk.Tk()
# root.title("人脸检测")
#
# # 创建选择视频文件按钮和文本框
# video_label = tk.Label(root, text="选择视频文件:")
# video_label.pack(pady=5)
# video_entry = tk.Entry(root, width=50)
# video_entry.pack(pady=5)
# select_button = tk.Button(root, text="选择文件", command=select_video)
# select_button.pack(pady=5)
#
# # 创建开始检测按钮
# detect_button = tk.Button(root, text="开始检测", command=start_detection)
# detect_button.pack(pady=10)
#
# # 创建退出按钮
# exit_button = tk.Button(root, text="退出", command=exit_app)
# exit_button.pack(pady=10)
#
# # 设置输出文件的初始序号
# output_index = 1
#
# # 运行主循环
# root.mainloop()
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog


def detect_faces(video_path, output_index, detection_method):
    if detection_method == "SSD":
        detect_faces_ssd(video_path, output_index)
    elif detection_method == "YOLOv3":
        detect_person_yolo(video_path, output_index)
    else:
        print("No detection method!")


def detect_faces_ssd(video_path, output_index):
    # Load the SSD model to detect faces
    modelFile = "res10_300x300_ssd_iter_140000.caffemodel"  # I download from "https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector"
    configFile = "deploy.prototxt"  # I download from "https://github.com/Shiva486/facial_recognition/blob/master/res10_300x300_ssd_iter_140000.caffemodel"

    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

    # Open the video source
    cap = cv2.VideoCapture(video_path)

    # Capture the frequency height and width
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create an object to store the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_filename = 'output{}_ssd.avi'.format(output_index)
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # Convert the frame to blob format
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Detect faces in the net
        net.setInput(blob)
        detections = net.forward()

        # Traverse all the output
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Draw the rectangle when the confidence > threshold
            if confidence > 0.5:
                # Get the location of the rectangle
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (x, y, w, h) = box.astype(int)

                # Draw the rectangle
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                text = "Confidence: {:.2f}".format(confidence)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('Detection Results - SSD', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # Write the frame
        out.write(frame)

    # Release
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("The output video is stored as {}".format(output_filename))


def detect_person_yolo(video_path, output_index):
    # Load the yolo3 model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # I download from "https://pjreddie.com/darknet/yolo/"

    # Load the label file
    with open("coco.names", "r") as f:  # I copy from "https://gitcode.com/pjreddie/darknet/blob/master/data/coco.names?raw=true%20-O%20./coco.names&utm_source=csdn_github_accelerator&isLogin=1"
        classes = f.read().strip().split("\n")

    # I only need person label
    person_index = classes.index("person")

    # Open the source video
    cap = cv2.VideoCapture(video_path)

    # Capture the frequency height and width
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create an object to store the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_filename = 'output{}_yolo3.avi'.format(output_index)
    out2 = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        height, width, channels = frame.shape

        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # Detecting objects using yolo3
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        # Detect person and locate its position
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == person_index:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Method to decrease the rectangles pointing to one person
        if boxes:
            # I tried several times to choose the suitable threshold
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)

            if indices is not None and len(indices) > 0:
                for i in indices.flatten():
                    box = boxes[i]
                    x, y, w, h = box[0], box[1], box[2], box[3]

                    # Drawing rectangles
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Write the processed frame to the output video file
        out2.write(frame)

        cv2.imshow('Detection Results - SSD', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out2.release()

    print("The output video is stored as {}".format(output_filename))


def select_video():
    # Open the window for user to choose video file
    filepath = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    video_entry.delete(0, tk.END)
    video_entry.insert(0, filepath)


def start_detection():
    # Decide which method to use
    video_path = video_entry.get()
    detection_method = detection_var.get()
    global output_index
    detect_faces(video_path, output_index, detection_method)
    output_index += 1


def exit_app():
    root.destroy()


# Create Tkinter window
root = tk.Tk()
root.title("Person Detection")

# Create the buttons and their captions
video_label = tk.Label(root, text="Choose video file :")
video_label.pack(pady=5)
video_entry = tk.Entry(root, width=50)
video_entry.pack(pady=5)
select_button = tk.Button(root, text="Select file", command=select_video)
select_button.pack(pady=5)

# Create a drop-down menu to select a detection method
detection_label = tk.Label(root, text="Select detection method:")
detection_label.pack(pady=5)
detection_var = tk.StringVar(root)
detection_var.set("SSD")  # Default DNN
detection_menu = tk.OptionMenu(root, detection_var, "SSD", "YOLOv3")
detection_menu.pack(pady=5)

# Create a button to start detection
detect_button = tk.Button(root, text="Start detection", command=start_detection)
detect_button.pack(pady=10)

# Create the quit button
exit_button = tk.Button(root, text="Quit", command=exit_app)
exit_button.pack(pady=10)

# Set the default file number: 1
output_index = 1

# run
root.mainloop()
