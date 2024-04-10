import requests
import json

base_url = "https://mrqhn4tot3.execute-api.ap-northeast-1.amazonaws.com/api/database/v1/"
# 獲取所有用戶列表
def get_all_users():
    url = f"{base_url}/list-users"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error getting user list: {response.status_code}")
        return None

# 獲取指定用戶的所有session資訊
def get_user_sessions(user_id):
    url = f"{base_url}/sessions/{user_id}?recent_first=true"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error getting user sessions: {response.status_code}")
        return None

# 獲取指定session的原始資料
def get_session_data(user_id, timestamp, macaddress):
    url = f"{base_url}/session-data/{user_id}/{timestamp}/{macaddress}?data_format=with_timestamp&delivery_type=full-on-demand&delivery_method=cached"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data_url = response.json()["url"]
        data_response = requests.get(data_url)
        if data_response.status_code == 200:
            return data_response.json()
        else:
            print(f"Error getting session data: {data_response.status_code}")
            return None
    else:
        print(f"Error getting session data URL: {response.status_code}")
        return None

# 主程式
def main():
    users = get_all_users()
    if users is not None:
        all_data = []
        for user in users:
            user_id = user["idusers"]
            sessions = get_user_sessions(user_id)
            if sessions is not None:
                for session in sessions["timestamp"]:
                    timestamp = session
                    macaddress = sessions["session_data"]["BLE_MAC_ADDRESS"][sessions["timestamp"].index(timestamp)]
                    session_data = get_session_data(user_id, timestamp, macaddress)
                    if session_data is not None:
                        session_info = {
                            "user_id": user_id,
                            "timestamp": timestamp,
                            "macaddress": macaddress,
                            "sample_rate": sessions["session_data"]["sample_rate"][sessions["timestamp"].index(timestamp)],
                            "session_note": sessions["session_data"]["session_notes"][sessions["timestamp"].index(timestamp)],
                            "raw_data": session_data["data"]["data"]
                        }
                        input(f"session_info: {session_info}")
                        all_data.append(session_info)
        
        # 將所有資料存成JSON檔案
        with open("all_data.json", "w") as file:
            json.dump(all_data, file)
        print("Data saved to all_data.json")
    else:
        print("Failed to retrieve data")

if __name__ == "__main__":
    main()