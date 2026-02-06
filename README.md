# USNSSS3000R - Raspberry Pi Waste Sorting System
## IoT-teknologi og Mikrokontrollere i Smarte Systemer  
## IoT-technology & Microcontrollers in Smart Systems
### Course project of University of South-Eastern Norway Campus Hønefoss

![Motion Sensor Active](assets/motion_sensor_active.svg)
![Developer Board](assets/developer_board.svg)
![Device Hub Icon](assets/device_hub_icon.svg)

### What is this?
An automated waste sorting system powered by machine learning image recognition. A Raspberry Pi with an integrated camera module classifies waste items and directs them to the appropriate recycling bin using a trained TensorFlow model.

### Why?
This project demonstrates practical applications of IoT technology and embedded systems in solving real-world environmental challenges. It combines hardware integration, machine learning, and edge computing—key skills in the Information Technology and Information Systems programme at University of South-Eastern Norway.

### Who is it for?
Course participants and lecturers assessing the project development. Others are welcome to fork and clone, but this is a student project with room for improvement. Use at your own risk.

### How to set up
Clone the repository and follow the setup instructions in the documentation folder (//TODO).

### How to use
Point the camera at a waste item. The system identifies the object and displays the appropriate bin classification.

### Potential improvements
- Optimize model inference speed for real-time performance on resource-constrained hardware
- Expand training dataset to improve classification accuracy for edge cases
- Implement a web interface or mobile app for monitoring and statistics
- Add feedback mechanism to continuously improve model accuracy
- Integrate physical actuators to automatically sort items into bins
- Implement proper error handling and logging for production deployment

### Why LiteRT?
We chose __LiteRT__ for on-device inference on the Raspberry Pi 4 Model B because:

- __Performance__:  
LiteRT delivers  up to 1.4x better the GPU performance than TensorFlow Lite and introduces NPU acceleration support

- __Cross-platform compatibility__:  
Supports Linux, Android, iOS, Windows, and Web - ensuring our model runs reliably across different devices

- __Lightweight deployment__:  
The `.tflite` model format is optimized for edge devices with limited computational resources, making it ideal for embedded systems like Raspberry Pi.

- __Unified workflow__:  
Simplifies deployment with consistent model conversion and inference across platforms

- __Future-proof__:  
LiteRT is Google's active framework for on-device AI, replacing the TensorFlow Lite with better support and features

This allows us to run real-time waste classification inference directly on the Raspberry Pi without relying on cloud services, ensuring privacy and low latency.

### Credits
Ambaya  
Thoresen  
Tvenning  
Villacorta
