无线表格检测
运行环境：torch110-cu113，地址：https://repo.hexops.cn/artifactory/ai-runtime/LORE-TSR.tar.gz

1.数据处理（table-tools文件夹）运行环境：paddlecls，地址：https://repo.hexops.cn/artifactory/ai-runtime/table_tools_paddlecls.tar.gz
  详细内容参考table-tools/readme.md，这里是把readme的步骤按照顺序逐个写入到单个脚本（如01_splitCell）中，
  可逐个执行，处理无限表格数据，进行训练集和验证集的划分，
  并将格式转换成lore-tsr所需格式，即最终无限表格数据
  脚本在autoPreMark中，对于转化中的格式可通过autoPreMark/vis_scripts 进行可视化，检查是否存在问题
  
- a.使用标注人员标注好的数据进行单元格切分，
    运行脚本及参数：
    cd table-tools  
    python autoPreMark/01_splitCell.py  
      image_folder_path: 图像文件夹路径(输入)  
      json_folder_path: 标签文件夹路径(输入)  
      output_folder：输出文件夹(输出)  

- b.根据单元格区域进行行列信息的划分 
    python autoPreMark/02_sureCellInfo.py  
      input_folder: 输入文件夹路径(输入)，a步骤结果输出作为b的输入  
      output_folder：输出文件夹(输出)  

- c.对单元格内多行文本进行合并，输入为原图和json识别标签，输出xml文件  
    python autoPreMark/03_json2xml.py  
      image_folder_path: 图像文件夹路径(输入)  
      json_folder_path: 识别标签文件夹路径(输入)  
      xml_output_folder: xml文件夹路径(输出)  

- d.融合行列信息和文本识别结果并排序，输入为b和c的输出结果，输出为json文件  
    python autoPreMark/04_mergeText.py   
      xmlsPath_folder: c的输出结果(输入)   
      json_folder_path: b的输出结果(输入)   
      res_path_folder: 融合后的json输出路径(输出)   

- e.训练集和测试集划分，指定划分比例，输入输出路径  
    python autoPreMark/5_0_split_datalabel.py   
      input_path: 融合后文件夹路径，里面应该包含原图像和d步骤的json 输出   
      output_path： 划分后的训练测试数据集路径  

- f.将划分好的训练测试集转化为lore-tsr格式  
    python autoPreMark/5_Convert_label.py   
      input_folder: e步骤划分好的训练集或者测试集数据   
      output_file: 转换后的文件名绝对路径，应当是path/to/train.json或者path/to/test.json   

2.训练脚本
  sh src/scripts/train/train_wireless.sh  
  --image_dir 数据集路径  
  --arch 模型结构  
  --load_model 加载模型路径  
  --load_processor 加载预处理模型路径  

3.推理脚本  
  sh src/scripts/infer/demo_wireless.sh  
  同训练脚本  

tip:
    1. 如果缺失dcn模块，导入项目目录路径export PYTHONPATH=/path/to/my_project   
    2. 训练脚本和测试脚本的参数要设置一致，如--arch  
  
