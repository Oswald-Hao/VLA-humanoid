# VLA Intent Message
# 识别出的用户意图

std_msgs/Header header
string session_id            # 会话ID
string intent_type           # 意图类型 (grasp, move, speak, etc.)
string intent_description    # 意图描述
float64 confidence           # 意图识别置信度
string[] entities            # 实体列表
string[] slots               # 槽位填充结果
string original_command      # 原始命令
string[] possible_actions    # 可能的动作列表
