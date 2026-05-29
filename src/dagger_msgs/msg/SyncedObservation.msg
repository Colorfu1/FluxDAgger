# Synchronized multi-camera and robot-state observation.

uint32 episode_id
uint32 step_id
bool   is_terminal

time   stamp

sensor_msgs/Image cam_high
sensor_msgs/Image cam_front
sensor_msgs/Image cam_left_wrist
sensor_msgs/Image cam_right_wrist

float32[] qpos
float32[] qvel
float32[] effort
float32[] base_vel
float32[] eepose

string   task_id
string   task_text

string   timestamps_json   # Per-stream timestamps encoded as JSON
