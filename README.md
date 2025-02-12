# AI ArmyVest - Thermal Vision System

## Overview

**AI ArmyVest** is a real-time thermal vision system designed for soldiers, enabling them to detect humans and anomalies in environments where human eyesight is ineffective, such as smoke-filled areas or complete darkness. A thermal camera attached to a soldier's vest captures the feed, which is processed using AI to detect objects and anomalies. The processed video stream is displayed on a small screen attached to the soldier's arm and can also be accessed remotely by Command and Control (C2) personnel in the control room.


![image](https://github.com/user-attachments/assets/f1f8506e-16d5-409d-8e32-92c6c6a6063c)



## Features

- **Real-time Thermal Vision**: Uses a thermal camera to detect heat signatures in low-visibility conditions.
- **AI-Powered Detection**: Employs a YOLO model to detect persons and anomalies.
- **Flask-based Streaming**: Streams the processed thermal feed to both the soldier's display and the control room via a Flask API.
- **Secure & Efficient Transmission**: Ensures efficient data transmission with optimized video encoding.

## Installation

### Prerequisites

- Python 3.x
- OpenCV
- NumPy
- Flask
- Ultralytics YOLO

### Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-repo/AI-ArmyVest.git
   cd AI-ArmyVest
   ```
   
2. **Install Dependencies**
    ```
    pip install -r requirements.txt
    ```

3. **Set up the model**

    - Place your trained YOLO model( `ThermVisionModel.pt` ) in the project directory.
    - Update the model path in `ThermalVsionV2.pt` :
        ```
        model = YOLO("path/to/ThermVisionModel.pt")
        ```    

4. **Run the Flask Server**
    ```
    python ThermalVisionV2.py
    ```
5. **Access the Video Stream**
    - Open a web browser and go to: 
        ```  
        http://<server-ip>:8002
        ```
    - The processed thermal video will be displayed.

## Usage

- The thermal camera captures video and sends it to the YOLO model for detection.
- The processed frames are streamed via Flask.
- The soldier views the stream on an arm-mounted display.
- C2 personnel can monitor the stream remotely.

  ![image](https://github.com/user-attachments/assets/dec53e0f-2e33-4c87-9ec2-92a7b66ab9ae)


## Future Enhancements

- Implement object tracking for better situational awareness.
- Enhance detection for additional anomaly types (e.g., hidden weapons, explosives).
- Improve streaming efficiency for lower latency.
