import gradio as gr
import cv2
import numpy as np
import os
import json
import time
from agent import Agent

# Proxy Sensor for Video Input
class VideoSensor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video {video_path}")
    
    def take_photo(self, save=False):
        ret, frame = self.cap.read()
        if not ret:
            # Loop the video for continuous demo
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                # Return blank if truly empty
                return np.zeros((480, 640, 3), dtype=np.uint8)
        return frame

    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

# State for browser-side webcam
class WebcamState:
    def __init__(self):
        self.prev_frame = None
        self.agent = None

webcam_state = WebcamState()

def run_agent_on_video(video_path):
    if not video_path:
        return
    sensor = VideoSensor(video_path)
    agent = Agent(sensor=sensor)
    for frame in agent.work():
        if frame is None: break
        # Convert BGR (cv2) to RGB (Gradio)
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        time.sleep(0.01) # Slight delay for streaming smoothness

def process_webcam_frame(frame):
    global webcam_state
    if frame is None:
        return None
    
    # Initialize agent on first frame if needed
    if webcam_state.agent is None:
        # We don't need a real sensor here as we pass frames manually
        webcam_state.agent = Agent(sensor=None)
    
    # Frame is RGB from Gradio, Agent expects BGR for processing
    current_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    if webcam_state.prev_frame is None:
        webcam_state.prev_frame = current_bgr.copy()
        return frame # Show raw on first frame
    
    # Run detection
    coco_bboxes = webcam_state.agent.predict(current_bgr, webcam_state.prev_frame)
    processed_bgr = webcam_state.agent.handle_prediction(current_bgr, webcam_state.prev_frame, coco_bboxes, 0.0)
    
    # Update state
    webcam_state.prev_frame = current_bgr.copy()
    
    # Return RGB for display
    return cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)

def clear_webcam_state():
    global webcam_state
    webcam_state.prev_frame = None
    webcam_state.agent = None

# Custom CSS
custom_css = """
.container { max-width: 1000px; margin: auto; padding-top: 20px; }
.header { text-align: center; margin-bottom: 2rem; color: #34495e; }
.tab-content { border-radius: 12px; margin-top: 15px; padding: 20px; border: 1px solid #ecf0f1; }
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🦉 Desktop Bird Detection Demo", elem_classes="header")
    gr.Markdown("Re-using the RPi Bird Detection Agent on desktop environments.")
    
    with gr.Tabs():
        with gr.TabItem("🎞️ Video File Input"):
            with gr.Column(elem_classes="tab-content"):
                gr.Markdown("### 📂 Upload and Process Video")
                video_input = gr.Video(label="Source Video")
                with gr.Row():
                    start_video_btn = gr.Button("▶️ Start Detection", variant="primary")
                    stop_video_btn = gr.Button("⏹️ Stop Detection", variant="stop")
                
                video_display = gr.Image(label="Processed Result")
                
                video_event = start_video_btn.click(
                    fn=run_agent_on_video,
                    inputs=video_input,
                    outputs=video_display
                )
                
                stop_video_btn.click(
                    fn=None,
                    inputs=None,
                    outputs=None,
                    cancels=[video_event]
                )

        with gr.TabItem("📷 Webcam Input"):
            with gr.Column(elem_classes="tab-content"):
                gr.Markdown("### 🎥 Live Browser-Webcam Stream")
                gr.Markdown("Click 'Capture' to allow webcam access in your browser.")
                
                webcam_input = gr.Image(sources=["webcam"], label="Webcam Input")
                webcam_display = gr.Image(label="Processed Result")
                
                # Each streaming frame from the browser calls this function
                webcam_input.stream(
                    fn=process_webcam_frame,
                    inputs=webcam_input,
                    outputs=webcam_display,
                    queue=False # Ensure low latency
                )
                
                clear_btn = gr.Button("🔄 Reset Agent State")
                clear_btn.click(fn=clear_webcam_state)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)
