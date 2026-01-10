# VLA Action Message
# 生成的结构化动作JSON

std_msgs/Header header
string session_id            # 会话ID
string action_id             # 动作ID
string action_type           # 动作类型
string json_data             # JSON格式的动作数据
float64 confidence           # 动作生成置信度
string[] required_skills    # 所需技能列表
string[] dependencies        # 依赖项
string execution_plan        # 执行计划
bool is_valid                # 动作是否有效
string error_message         # 错误信息（如果有）
