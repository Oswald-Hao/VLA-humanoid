# VLA Object Depth Message
# VLA物体深度信息消息

Header header
string object_id                      # 物体唯一标识符
string object_class                    # 物体类别
string object_name                     # 物体名称
geometry_msgs/Point3D position         # 物体3D位置
geometry_msgs/Vector3 size             # 物体3D尺寸
float32 distance                       # 到相机的距离
float32 depth_confidence               # 深度估计置信度
string depth_source                    # 深度信息来源
time timestamp                        # 时间戳
bool is_valid                          # 深度信息是否有效
float32 point_density                  # 点云密度
string reference_frame                 # 参考坐标系
