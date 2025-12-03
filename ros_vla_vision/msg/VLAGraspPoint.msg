# VLA Grasp Point Message
# VLA抓取点消息

Header header
string object_id                      # 物体唯一标识符
string object_class                    # 物体类别
string object_name                     # 物体名称
geometry_msgs/Pose grasp_pose          # 抓取位姿
float32 grasp_quality                  # 抓取质量评分
float32 grasp_width                    # 抓取宽度
string grasp_type                      # 抓取类型 (pinch, power, etc.)
geometry_msgs/Vector3 approach_vector # 接近向量
geometry_msgs/Vector3 grasp_normal     # 抓取法向量
float32 confidence                     # 抓取点置信度
time timestamp                        # 时间戳
bool is_feasible                       # 抓取是否可行
string gripper_type                    # 适配的夹爪类型
string reference_frame                 # 参考坐标系
