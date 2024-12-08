import json
from datetime import datetime
import os

class UserActivityTracker:
    def __init__(self):
        self.log_dir = 'logs'
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def log_activity(self, user_id, action, details=None):
        timestamp = datetime.now().isoformat()
        log_entry = {
            'timestamp': timestamp,
            'user_id': user_id,
            'action': action,
            'details': details or {}
        }
        
        log_file = os.path.join(self.log_dir, f'user_activity_{datetime.now().strftime("%Y%m%d")}.json')
        
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Error logging activity: {e}")
    
    def get_user_activities(self, user_id, date=None):
        activities = []
        
        if date is None:
            date = datetime.now().strftime("%Y%m%d")
        
        log_file = os.path.join(self.log_dir, f'user_activity_{date}.json')
        
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    for line in f:
                        activity = json.loads(line)
                        if activity['user_id'] == user_id:
                            activities.append(activity)
        except Exception as e:
            print(f"Error retrieving activities: {e}")
        
        return activities