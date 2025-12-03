# VLA VLM Result Message
# VLA视觉语言模型结果消息

Header header
string task_id                         # 任务ID
string instruction                     # 用户指令
string task_type                       # 任务类型 (vision_only, grasping, location_query)
string scene_description               # 场景描述
string scene_type                      # 场景类型
string[] objects                       # 检测到的物体列表
string[] scene_tags                    # 场景标签
float32 confidence                     # 总体置信度
float32 processing_time               # 处理时间
time timestamp                        # 时间戳

# 物体详细信息（如果涉及抓取或位置查询）
VLAObjectPose[] object_poses          # 物体位姿信息
VLAObjectDepth[] object_depths        # 物体深度信息
VLAGraspPoint[] grasp_points          # 抓取点信息

# 任务相关标志
bool include_pose                      # 是否包含位姿信息
bool include_depth                     # 是否包含深度信息
bool include_grasp_points             # 是否包含抓取点信息
bool is_grasping_task                 # 是否为抓取任务
bool is_location_query                # 是否为位置查询任务

# 目标物体信息（从指令中提取）
string target_object                   # 目标物体名称
string target_object_class            # 目标物体类别
bool target_object_found              # 是否找到目标物体

# VLM API信息
string vlm_model                      # 使用的VLM模型
string api_response_summary           # API响应摘要
bool api_call_success                 # API调用是否成功
string error_message                  # 错误信息（如果有）
