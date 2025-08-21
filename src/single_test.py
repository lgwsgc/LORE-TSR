import os
from opts import opts
opt = opts().init()
import cv2
from yolov5_onnx import ONNXModelYolov5
from lib.detectors.detector_factory import detector_factory
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

opt.debug = 1 
opt.stacking_layers = 4
opt.tsfm_layers = 4
opt.K = 3000
opt.MK= 5000
opt.vis_thresh = 0.20
opt.scores_thresh = 0.2
opt.anno_path == ''
opt.wiz_detect = True
# opt.load_model = "../model/ckpt_wtw/model_best.pth"
# opt.load_processor = "../model/ckpt_wtw/processor_best.pth"

opt.load_model = "../model/ckpt_wireless/model_best.pth"
opt.load_processor = "../model/ckpt_wireless/processor_best.pth"

opt.demo_name = "demo_wireless"
Detector = detector_factory[opt.task]
detector = Detector(opt)
opt.demo = "/ssdata/user/zxy/code/lore/LORE-TSR/data"
# image_path = "../data/test/test"
opt.output_dir = "./data/log"

onnx_path = "/ssdata/user/zxy/code/lore/LORE-TSR/model/table_detect/tabledetect_20231102_00.onnx"
table_detect = ONNXModelYolov5(onnx_path,640,confThreshold=0.5, nmsThreshold=0.5)
table_path = "/ssdata/user/zxy/code/lore/LORE-TSR/data/table_detect"

# if not os.path.exists(table_path)
if not os.path.exists(opt.output_dir):
      os.mkdir(opt.output_dir)
if not os.path.exists(os.path.join("./data",'vis')):
      os.mkdir(os.path.join("./data",'vis'))
for dir in os.listdir(opt.demo):
      dir_path = os.path.join(opt.demo,dir)
      if dir !="test":
            continue
      for file in os.listdir(dir_path):
            # if file !="0003_20210527143057_3517f483-0fd7-42d9-8bf8-fb277b5efcc8.jpg":
            #       continue
            filepath = os.path.join(dir_path,file)
            last_image = cv2.imread(filepath)
            if last_image is None:
                  continue
            ret = detector.run(opt,filepath,None)
            # detection = table_detect.detect(last_image)
            # if detection:
            #       for index,s in  enumerate(detection):
            #             image = last_image[s[1]:s[3],s[0]:s[2]]
            #             image_path = os.path.join(table_path,str(dir)+"_"+file.split(".")[0]+str(index)+"_"+".jpg")
            #             cv2.imwrite(image_path,image)
                       
            #             # print(ret["results"])
                       
