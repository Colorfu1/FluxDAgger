# Per-arm runtime status published by arm_node.

string arm_name
string current_mode        # "master" or "slave"
string subscribed_topic    # Empty when unsubscribed
bool is_enabled
bool is_ready              # SDK connection status
