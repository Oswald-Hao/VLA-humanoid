# VLA Object Pose Message
# VLA物体位姿消息

Header header
string object_id                      # 物体唯一标识符
string object_class                    # 物体类别
string object_name                     # 物体名称
geometry_msgs/Pose pose                # 物体6D位姿
float32 confidence                     # 位姿估计置信度
string detection_method                # 检测方法
time timestamp                        # 时间戳
bool is_tracked                        # 是否被跟踪
float32 tracking_quality               # 跟踪质量
string reference_frame                 # 参考坐标系
