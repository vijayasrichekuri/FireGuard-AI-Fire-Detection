import requests
import os

def download_sounds():
    # Create sounds directory
    os.makedirs("sounds", exist_ok=True)
    
    # Sound URLs
    sounds = {
        "fire_alarm.mp3": "https://assets.mixkit.co/sfx/download/mixkit-fire-engine-siren-1995.mp3",
        "success.mp3": "https://assets.mixkit.co/sfx/download/mixkit-correct-answer-tone-2870.mp3"
    }
    
    for filename, url in sounds.items():
        if not os.path.exists(f"sounds/{filename}"):
            print(f"Downloading {filename}...")
            response = requests.get(url)
            with open(f"sounds/{filename}", "wb") as f:
                f.write(response.content)
            print(f"✅ Downloaded {filename}")
        else:
            print(f"✅ {filename} already exists")

# Run this once
download_sounds()
