# Collector request for operator input from the controller.

string request_id
string prompt
string valid_chars         # e.g. "01" accepts only 0 or 1
float32 timeout            # Seconds; -1 means wait forever
